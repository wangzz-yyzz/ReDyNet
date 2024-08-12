import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseModel(nn.Module, ABC):
    def __init__(self, data_feature):
        super().__init__()
        self.total_dim = data_feature['feature_dim']
        self.output_dim = data_feature.get('output_dim')
        self.node_num = data_feature['num_nodes']
        self.ext_dim = data_feature.get('ext_dim')

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self._scaler = data_feature.get('scaler')

    @abstractmethod
    def calculate_loss_train(self, batch):
        # y_true = batch['y']
        # y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        # y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        pass

    @abstractmethod
    def predict(self, batch):
        # x = batch['X']  # [B,T,N,C]
        pass

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
