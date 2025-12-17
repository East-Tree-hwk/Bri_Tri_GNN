import torch
import torch.nn.functional as F
from torch import nn


class Tri_GNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 顺时针
        self.h_hidden = nn.Linear(5, 20)
        self.m_hidden = nn.Linear(5, 20)
        self.t_hidden = nn.Linear(5, 20)

        self.weights = nn.Parameter(torch.tensor([0.4, 0.2, 0.4]))

        self.edge_layers = nn.Sequential(
            nn.Linear(20, 40),
            nn.BatchNorm1d(40),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.1),

            nn.Linear(40, 60),
            nn.BatchNorm1d(60),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.1),

            nn.Linear(60, 80),
            nn.BatchNorm1d(80),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.1),

            nn.Linear(80, 42),
        )

    def forward(self, x):
        x1 = x[:, :5]
        x2 = x[:, 10:]
        x3 = x[:, 5:10]

        alp_h, alp_m, alp_t = self.weight_factor(x1, x2, x3)

        xh = self.layer_norm_h(F.leaky_relu(self.h_hidden(x1)))
        xm = self.layer_norm_m(F.leaky_relu(self.m_hidden(x2)))
        xt = self.layer_norm_t(F.leaky_relu(self.t_hidden(x3)))

        weights = F.softmax(self.weights, dim=0)
        x = weights[0] * alp_h * xh + weights[1] * alp_m * xm + weights[2] * alp_t * xt

        x = self.edge_layers(x)

        return x

    def weight_factor(self, x1, x2, x3):
        x_h = torch.diag(torch.mm(x1, x1.T))
        x_m = torch.diag(torch.mm(x3, x3.T))
        x_t = torch.diag(torch.mm(x2, x2.T))
        x_total = x_h + x_m + x_t
        alp_h = x_h / x_total
        alp_h = alp_h.unsqueeze(1)
        alp_m = x_m / x_total
        alp_m = alp_m.unsqueeze(1)
        alp_t = x_t / x_total
        alp_t = alp_t.unsqueeze(1)
        return alp_h, alp_m, alp_t
