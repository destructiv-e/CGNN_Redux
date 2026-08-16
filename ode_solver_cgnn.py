import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint

class ODEFunc(nn.Module):
    def __init__(self, in_features, out_features, opt, adj, deg):
        super().__init__()
        self.opt = opt
        self.adj = adj
        self.x0 = None
        self.nfe = 0
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = opt['alpha']
        self.alpha_train = nn.Parameter(self.alpha * torch.ones(adj.shape[1]))
        self.w = nn.Parameter(torch.eye(opt['hidden_dim']))
        self.d = nn.Parameter(torch.zeros(opt['hidden_dim']) + 1)

    def forward(self, t, x):
        self.nfe += 1
        alph = F.sigmoid(self.alpha_train).unsqueeze(dim=1)
        ax = torch.spmm(self.adj, x)
        return alph * 0.5 * (ax - x) + self.x0

class ODEblock(nn.Module):
    def __init__(self, odefunc, t=torch.tensor([0, 1])):
        super().__init__()
        self.t = t
        self.odefunc = odefunc
        self.nfe = 0

    def set_x0(self, x0):
        self.odefunc.x0 = x0.clone().detach()

    def forward(self, x):
        self.nfe += 1
        t = self.t.type_as(x)
        return odeint(self.odefunc, x, t)[1]

    def __repr__(self):
        time_range = f"{self.t[0].item()} -> {self.t[1].item()}"
        return f"{self.__class__.__name__}(Time Interval {time_range})"