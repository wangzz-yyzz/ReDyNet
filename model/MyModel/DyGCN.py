import torch
import torch.nn as nn
import torch.nn.functional as F


class DyGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheby_k, embed_dim, aggregate_type='sum'):
        super(DyGCN, self).__init__()
        self.cheby_k = cheby_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheby_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        if aggregate_type == 'weighted_sum':
            self.weights_cheby = torch.nn.Parameter(torch.ones(cheby_k))
        self.aggregate_type = aggregate_type

    def forward(self, x, all_emb, station_emb, return_supports=False):
        batch_size, node_num, _ = all_emb.shape
        supports = F.softmax(F.relu(torch.matmul(all_emb, all_emb.transpose(1, 2))), dim=-1)  # [B, N, N]
        t_k_0 = torch.eye(node_num).to(supports.device)  # [B, N, N]
        t_k_0 = t_k_0.unsqueeze(0).expand(batch_size, -1, -1)  # [B, N, N]
        support_set = [t_k_0, supports]
        for k in range(2, self.cheby_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports_cheby = torch.stack(support_set, dim=0)  # [cheby_k, B, N, N]
        supports_cheby = supports_cheby.permute(1, 0, 2, 3)  # [B, cheby_k, N, N]

        # B, N, cheby_k, dim_in, dim_out
        weights = torch.einsum('bni,ikop->bnkop', station_emb, self.weights_pool)
        # B, N, dim_out
        bias = torch.matmul(station_emb, self.bias_pool)
        # B, cheby_k, N, dim_in
        x_g = torch.einsum('bkij,bjd->bkid', supports_cheby, x)
        # B, N, cheby_k, dim_out
        x_g_conv = torch.einsum('bkni,bnkio->bnko', x_g, weights)
        # B, N, dim_out
        if self.aggregate_type == 'sum':
            x_g_conv = x_g_conv.sum(dim=2) + bias
        elif self.aggregate_type == 'weighted_sum':
            x_g_conv = x_g_conv * self.weights_cheby.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            x_g_conv = x_g_conv.sum(dim=2) + bias

        if return_supports:
            return x_g_conv, supports
        return x_g_conv


class MutilHeadDyGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheby_k, embed_dim, num_heads, aggregate_type='sum'):
        super(MutilHeadDyGCN, self).__init__()
        self.dygcn_list = nn.ModuleList([DyGCN(dim_in, dim_out // num_heads, cheby_k, embed_dim, aggregate_type)
                                         for _ in range(num_heads)])
        self.num_heads = num_heads

    def forward(self, x, all_emb, station_emb, return_supports=False):
        head_outs = []
        for i in range(self.num_heads):
            head_outs.append(self.dygcn_list[i](x, all_emb, station_emb))

        if return_supports:
            return torch.cat(head_outs, dim=-1), torch.Tensor(0).to(x.device)
        return torch.cat(head_outs, dim=-1)
