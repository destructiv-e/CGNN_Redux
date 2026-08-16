from encoders import *
from ode_solvers.ode_solver_cgnn import *

class GNN(nn.Module):
    def __init__(self, opt, adj, deg, time):
        super().__init__()
        self.opt = opt
        self.adj = adj
        self.deg = deg
        self.T = time

        self._init_encoder()
        self._init_ode_block()
        self._init_classifier()

        if opt['cpu']:
            self.cpu()

    def _init_encoder(self):
        encoder_type = self.opt.get('encoder_type', 'with_dropout')

        if encoder_type == 'with_dropout':
            self.encoder = EncoderWithDropout(
                self.opt['num_feature'],
                self.opt['hidden_dim'],
                self.opt['input_dropout']
            )
        elif encoder_type == 'without_dropout':
            self.encoder = EncoderWithoutDropout(
                self.opt['num_feature'],
                self.opt['hidden_dim']
            )
        elif encoder_type == 'mlp':
            self.encoder = EncoderMLP(
                self.opt['num_feature'],
                self.opt['hidden_dim']
            )

    def _init_ode_block(self):
        self.odeblock = ODEblock(
            ODEFunc(
                2 * self.opt['hidden_dim'],
                2 * self.opt['hidden_dim'],
                self.opt,
                self.adj,
                self.deg
            ),
            t=torch.tensor([0, self.T])
        )

    def _init_classifier(self):
        self.m2 = nn.Linear(self.opt['hidden_dim'], self.opt['num_class'])

    def reset(self):
        self.encoder = nn.Linear(self.opt['num_feature'], self.opt['hidden_dim'])
        self.m2.reset_parameters()

    def forward(self, x):
        x = self.encoder(x)
        c_aux = torch.zeros(x.shape).cpu()
        x = torch.cat([x, c_aux], dim=1)
        self.odeblock.set_x0(x)

        z = self.odeblock(x)
        z = torch.split(z, x.shape[1] // 2, dim=1)[0]
        z = F.relu(z)
        z = F.dropout(z, self.opt['dropout'], training=self.training)

        return self.m2(z)