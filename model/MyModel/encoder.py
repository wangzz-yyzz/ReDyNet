import torch
import torch.nn as nn
import torch.nn.functional as F

from model.MyModel.collector import Collector

collectors = Collector('saved')


class StationEncoder(nn.Module):
    def __init__(self, num_sites, encoder_dim, station_embed_dim, input_time):
        super(StationEncoder, self).__init__()
        self.input_time = input_time
        self.site_embedding = nn.Embedding(num_sites, station_embed_dim)
        self.flow_encoder = nn.Linear(input_time * 2, encoder_dim)
        self.embed_encoder = nn.Linear(station_embed_dim, encoder_dim)
        self.station_embed_dim = station_embed_dim
        self.encoder_dim = encoder_dim
        self.dropout = nn.Dropout(0.1)

    def forward(self, flow):
        # x: [B, T, N, 2]
        batch_size, time_len, num_sites, features = flow.shape

        site_indices = torch.arange(num_sites).to(flow.device)
        site_indices = site_indices.unsqueeze(0).expand(batch_size, num_sites)

        site_embeds = self.site_embedding(site_indices)
        site_embeds = self.dropout(site_embeds)  # [B, N, embed_dim]

        flow = flow.permute(0, 2, 1, 3).reshape(batch_size, num_sites, time_len * features)  # [B, N, T*2]
        flow = self.flow_encoder(flow)  # [B, N, output_dim]

        site_features = self.embed_encoder(site_embeds)  # [B, N, output_dim]

        p = F.sigmoid(site_features)
        q = F.tanh(flow)

        station_flow_feature = p * q + (1 - p) * site_features  # [B, N, output_dim]
        station_flow_feature = station_flow_feature.view(batch_size, num_sites, self.encoder_dim)  # [B, N, output_dim]

        return station_flow_feature


class DateEncoder(nn.Module):
    def __init__(self, encoder_dim, date_embed_dim, input_time):
        super(DateEncoder, self).__init__()
        self.input_time = input_time
        self.hour_embedding = nn.Embedding(24, date_embed_dim)
        self.weekday_embedding = nn.Embedding(7, date_embed_dim)
        self.flow_encoder = nn.Linear(input_time * 2, encoder_dim)
        self.embed_encoder = nn.Linear(input_time * date_embed_dim, encoder_dim)
        self.date_embed_dim = date_embed_dim
        self.encoder_dim = encoder_dim
        self.dropout = nn.Dropout(0.1)

    def forward(self, flow, date):
        # x:  [B, T, N, 2]
        batch_size, time_len, num_sites, features = flow.shape
        # date: [B, T, N, 2]
        hour = date[..., 0]  # [B, T, N]
        weekday = date[..., 1]  # [B, T, N]
        hour = hour.long()
        weekday = weekday.long()

        hour_embeds = self.hour_embedding(hour)  # [B, T, N, embed_dim]
        weekday_embeds = self.weekday_embedding(weekday)  # [B, T, N, embed_dim]
        date_embeds = hour_embeds + weekday_embeds

        date_embeds = date_embeds.permute(0, 2, 1, 3).reshape(batch_size, num_sites, time_len * self.date_embed_dim)
        date_embeds = self.dropout(date_embeds)

        flow = flow.permute(0, 2, 1, 3).reshape(batch_size, num_sites, time_len * features)  # [B, N, T*2]
        flow = self.flow_encoder(flow)  # [N, B, output_dim]

        date_feature = self.embed_encoder(date_embeds)  # [B, N, output_dim]

        p = F.sigmoid(date_feature)
        q = F.tanh(flow)

        date_flow_feature = p * q + (1 - p) * date_feature  # [B, N, output_dim]
        date_flow_feature = date_flow_feature.view(batch_size, num_sites, self.encoder_dim)  # [B, N, output_dim]

        return date_flow_feature


class EnvEncoder(nn.Module):
    def __init__(self, encoder_dim, input_time):
        super(EnvEncoder, self).__init__()
        self.input_time = input_time
        self.flow_encoder = nn.Linear(input_time * 2, encoder_dim)
        self.env_encoder = nn.Linear(input_time * 5, encoder_dim)
        self.encoder_dim = encoder_dim
        self.dropout = nn.Dropout(0.1)

    def forward(self, flow, env):
        # x: [B, T, N, 2]
        batch_size, time_len, num_sites, features = flow.shape
        _, _, _, env_features = env.shape

        flow = flow.permute(0, 2, 1, 3).reshape(batch_size, num_sites, time_len * features)  # [B, N, T*2]
        flow = self.flow_encoder(flow)  # [B, N, output_dim]

        env = env.permute(0, 2, 1, 3).reshape(batch_size, num_sites, time_len * env_features)  # [B, N, T*5]
        env = self.dropout(env)

        env_feature = self.env_encoder(env)  # [B, N, output_dim]

        p = F.sigmoid(env_feature)
        q = F.tanh(flow)

        enc_flow_feature = p * q + (1 - p) * env_feature
        enc_flow_feature = enc_flow_feature.view(batch_size, num_sites, self.encoder_dim)  # [B, N, output_dim]

        return enc_flow_feature
