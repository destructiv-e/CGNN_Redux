import torch
import torch.nn as nn
import torch.nn.functional as F
from coder_encoder_cgnn_wcgnn import *
from ode_solver_wcgnn import *

class WGNN(nn.Module):
    def __init__(self, opt, adj, deg, time):
        super().__init__()
        self.opt = opt
        self.adj = adj
        self.T = time

        self.encoder = self._get_encoder()
        self.odeblock = ODEblockW(
            ODEFuncW(2 * opt['hidden_dim'], 2 * opt['hidden_dim'], opt, adj, deg),
            t=torch.tensor([0, self.T])
        )
        self.classifier = nn.Linear(opt['hidden_dim'], opt['num_class'])

        if opt['cpu']:
            self.cpu()

    def _get_encoder(self):
        encoder_type = self.opt.get('encoder_type', 'with_dropout')
        if encoder_type == 'with_dropout':
            return EncoderWithDropout(
                self.opt['num_feature'],
                self.opt['hidden_dim'],
                self.opt['input_dropout']
            )
        elif encoder_type == 'without_dropout':
            return EncoderWithoutDropout(
                self.opt['num_feature'],
                self.opt['hidden_dim']
            )
        elif encoder_type == 'mlp':
            return EncoderMLP(
                self.opt['num_feature'],
                self.opt['hidden_dim']
            )
        raise ValueError(f"Unknown encoder type: {encoder_type}")

    def reset(self):
        self.encoder = nn.Linear(self.opt['num_feature'], self.opt['hidden_dim'])
        self.classifier.reset_parameters()

    def forward(self, x):
        x = self.encoder(x)
        c_aux = torch.zeros(x.shape).cpu()
        x = torch.cat([x, c_aux], dim=1)
        self.odeblock.set_x0(x)

        z = self.odeblock(x)
        z = torch.split(z, x.shape[1] // 2, dim=1)[0]
        z = F.relu(z)
        z = F.dropout(z, self.opt['dropout'], training=self.training)

        return self.classifier(z)