import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import BatchNorm
import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self,input_dim,output_dim):
        super(GCN, self).__init__()

        self.GConv1 = GCNConv(input_dim,1024)
        self.bn1 = BatchNorm(1024)

        self.GConv2 = GCNConv(1024,1024)
        self.bn2 = BatchNorm(1024)

        self.linear = nn.Sequential(
            nn.Linear(1024,256),
            nn.ReLU(inplace=True),
            nn.Linear(256,output_dim)
        )

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        # u_s_q = data.u_mul_s
        # v_q = data.svd_v

        self.f1 = x  # 原始特征(时域输入时为时域特征，频域输入时为频域特征)

        x = self.GConv1(x, edge_index, edge_weight)
        x = self.bn1(x)
        x = F.relu(x)

        # x = u_s_q @ v_q.T @ x

        self.f2 = x  # 第一层特征(嵌入特征）

        x = self.GConv2(x, edge_index, edge_weight)
        x = self.bn2(x)
        x = F.relu(x)

        # x = u_s_q @ v_q.T @ x

        self.f3 = x  # 第二层特征

        x = self.linear(x)

        self.f4 = x  # 全连接层特征

        out = F.log_softmax(x, dim=1)

        loss_reg = 0
        for param in self.parameters():
            loss_reg += param.norm(2).square()
        loss_reg *= 1e-7
        #
        return out

    def get_fea(self):
        return [self.f1,self.f2,self.f3,self.f4]