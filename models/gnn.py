"""Модель CGNN: GNN как непрерывная динамическая система на графе.

Архитектура: энкодер признаков -> ODE-блок (непрерывная диффузия по графу,
см. `ode_solvers.ode_solver_cgnn`) -> линейный классификатор.
"""

from encoders import *
from ode_solvers.ode_solver_cgnn import *


class GNN(nn.Module):
    """CGNN: encoder -> ODEblock (диффузия по графу) -> классификатор."""

    def __init__(self, opt, adj, deg, time):
        """
        Args:
            opt: словарь гиперпараметров (см. `trainers.train` для полного списка).
            adj: разреженная матрица смежности графа.
            deg: степени узлов (передаются в ODE-блок).
            time: конечное время интегрирования ODE (T).
        """
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
        """Создаёт энкодер входных признаков по типу из `opt['encoder_type']`."""
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
        """Создаёт ODE-блок диффузии по графу на интервале [0, T].

        Скрытая размерность ODE-блока в 2 раза больше `hidden_dim`, так как
        в `forward` к закодированным признакам конкатенируется вспомогательный
        нулевой блок `c_aux` той же размерности (см. комментарий в `forward`).
        """
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
        """Создаёт линейный классификатор над скрытым представлением узла."""
        self.m2 = nn.Linear(self.opt['hidden_dim'], self.opt['num_class'])

    def reset(self):
        """Сбрасывает энкодер и классификатор к начальному состоянию."""
        self.encoder = nn.Linear(self.opt['num_feature'], self.opt['hidden_dim'])
        self.m2.reset_parameters()

    def forward(self, x):
        """
        Args:
            x: признаки узлов [num_nodes x num_feature].

        Returns:
            torch.Tensor: логиты классов [num_nodes x num_class].
        """
        x = self.encoder(x)
        # расширяем представление вспомогательным нулевым блоком той же
        # размерности -- ODE-блок работает в удвоенной размерности, а после
        # интегрирования обратно берётся только первая половина (см. ниже)
        c_aux = torch.zeros(x.shape).cpu()
        x = torch.cat([x, c_aux], dim=1)
        self.odeblock.set_x0(x)

        z = self.odeblock(x)
        z = torch.split(z, x.shape[1] // 2, dim=1)[0]  # берём только "основную" половину
        z = F.relu(z)
        z = F.dropout(z, self.opt['dropout'], training=self.training)

        return self.m2(z)
