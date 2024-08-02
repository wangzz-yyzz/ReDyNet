import torch.nn as nn


class GTUnit(nn.Module):
    def __init__(self, input_dim):
        super(GTUnit, self).__init__()
        self.input_dim = input_dim
        self.gate = nn.Linear(input_dim, input_dim)
        self.update = nn.Linear(input_dim, input_dim)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        p = self.sigmoid(self.gate(x))
        q = self.tanh(self.update(x))
        h = p * q + (1 - p) * x
        return h


class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gtu = GTUnit(input_dim)
        self.fc = nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, flow):
        batch_size, num_sites, features = flow.shape
        flow = self.gtu(flow)
        flow = self.fc(flow)
        return flow


class VaeEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(VaeEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gtu = GTUnit(input_dim)
        self.fc = nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, flow):
        batch_size, num_sites, features = flow.shape
        flow = self.gtu(flow)
        flow = self.fc(flow)
        return flow


class VaeDecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(VaeDecoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gtu = GTUnit(input_dim)
        self.fc = nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, flow):
        batch_size, num_sites, features = flow.shape
        flow = self.gtu(flow)
        flow = self.fc(flow)
        return flow
