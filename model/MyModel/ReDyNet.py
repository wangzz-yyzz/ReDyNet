import nni
import torch
import torch.nn as nn

import loss
from model.BaseModel import BaseModel
from model.MyModel.DyGCN import DyGCN
from model.MyModel.attention import SelfAttention
from model.MyModel.collector import Collector
from model.MyModel.decoder import Decoder, VaeDecoder, VaeEncoder
from model.MyModel.encoder import StationEncoder, DateEncoder, EnvEncoder

collectors = Collector('saved')

class ReDyNet(BaseModel):
    def __init__(self, config, data_feature):
        super().__init__(data_feature)
        self.total_dim = data_feature['feature_dim']
        self.output_dim = data_feature.get('output_dim')
        self.input_time = config['input_time']
        self.output_time = config['output_time']
        self.node_num = data_feature['num_nodes']

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self._scaler = data_feature.get('scaler')

        self.ext_dim = data_feature.get('ext_dim')
        self.gcn_dim = config['gcn_dim']
        self.encoder_dim = config['encoder_dim']
        self.station_embed_dim = config['station_embed_dim']
        self.date_embed_dim = config['date_embed_dim']
        self.vae_dim = config['vae_dim']

        self.cheby_k = config['cheby_k']
        self.a = config['a']
        self.temperature = config['temperature']
        self.beta = config['beta']
        self.vae_ratio = config['vae_ratio']

        self.station_encoder = StationEncoder(num_sites=self.node_num, encoder_dim=self.encoder_dim,
                                              station_embed_dim=self.station_embed_dim,
                                              input_time=self.input_time)
        self.date_encoder = DateEncoder(encoder_dim=self.encoder_dim, date_embed_dim=self.date_embed_dim,
                                        input_time=self.input_time)
        self.env_encoder = EnvEncoder(encoder_dim=self.encoder_dim, input_time=self.input_time)

        self.dygcn = DyGCN(dim_in=self.input_time * self.output_dim, dim_out=self.gcn_dim, cheby_k=self.cheby_k,
                           embed_dim=self.encoder_dim)

        self.decoder = Decoder(input_dim=self.gcn_dim, output_dim=self.output_dim * self.output_time)

        self.gcn_activation = nn.GELU()

        self.union_norm = nn.LayerNorm(self.encoder_dim)
        self.station_norm = nn.LayerNorm(self.encoder_dim)
        self.date_norm = nn.LayerNorm(self.encoder_dim)
        self.env_norm = nn.LayerNorm(self.encoder_dim)

        self.self_attention = SelfAttention(self.encoder_dim, self.encoder_dim, self.encoder_dim)

        self.vae_encoder = VaeEncoder(input_dim=self.encoder_dim, output_dim=self.vae_dim)
        self.rec_decoder = VaeDecoder(input_dim=self.vae_dim, output_dim=self.encoder_dim)
        self.fc_mu = nn.Linear(self.vae_dim, self.vae_dim)
        self.fc_log_var = nn.Linear(self.vae_dim, self.vae_dim)
        self.rec_layer_norm = nn.LayerNorm(self.encoder_dim)

        self._init_parameters()

    def forward(self, batch, return_supports=False):
        x = batch['X']  # [B,T,N,C]

        flow = x[..., :self.output_dim]  # [B,T,N,2]
        time = x[..., self.output_dim:self.output_dim + 2]  # [B,T,N,2]
        env = x[..., self.output_dim + 2:]  # [B,T,N,5]

        flow_station = self.station_encoder(flow)  # [B,N,64]
        flow_date = self.date_encoder(flow, time)  # [B,N,64]
        flow_env = self.env_encoder(flow, env)  # [B,N,64]

        union_origin = flow_station + flow_date + flow_env  # [B,N,64]
        union_origin = self.union_norm(union_origin)  # [B,N,64]

        z = self.vae_encoder(union_origin)  # [B,N,64]
        mu = self.fc_mu(z)  # [B,N,64]
        log_var = self.fc_log_var(z)  # [B,N,64]
        z = self.reparameterize(mu, log_var)  # [B,N,64]
        rec = self.rec_decoder(z)  # [B,N,64]
        union = union_origin - rec  # [B,N,64]
        union = self.rec_layer_norm(union)  # [B,N,64]

        flow_station = self.station_norm(flow_station)
        flow_date = self.date_norm(flow_date)
        flow_env = self.env_norm(flow_env)

        flow = flow.permute(0, 2, 1, 3)  # [B,N,T,C]
        flow = flow.reshape(flow.shape[0], flow.shape[1], -1)  # [B,N,T*C]

        gcn_output = self.dygcn(flow, union, flow_station, return_supports=False)  # [B,N,64]
        gcn_output = self.gcn_activation(gcn_output)  # [B,N,64]

        output = self.decoder(gcn_output)  # [B,N,4*2]
        output = output.reshape(output.shape[0], output.shape[1], self.output_time, self.output_dim)  # [B,N,T,C]
        output = output.permute(0, 2, 1, 3)  # [B,T,N,C]

        return (output, flow_station, flow_date, flow_env,
                {'mu': mu, 'log_var': log_var, 'rec_union': rec, 'union': union_origin})

    def calculate_loss_train(self, batch):
        y_true = batch['y']
        y_predicted, flow_station, flow_date, flow_env, vae = self.forward(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        loss_pred = loss.huber_loss_torch(y_predicted, y_true)

        mu, log_var, rec, union = vae['mu'], vae['log_var'], vae['rec_union'], vae['union']
        bce = nn.BCEWithLogitsLoss()(rec, union)
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        vae_loss = bce + self.beta * kld

        return loss_pred + vae_loss * self.vae_ratio

    def predict(self, batch):
        output, _, _, _, _ = self.forward(batch)
        return output

    def reparameterize(self, mu, log_var):
        std = log_var.mul(0.5).exp_()
        eps = std.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def _init_parameters(self):
        print('Initializing parameters...')
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
