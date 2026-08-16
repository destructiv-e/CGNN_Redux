"""Модель WCGNN: CGNN с дополнительным обучаемым взаимодействием признаков.

Архитектура: энкодер признаков -> ODE-блок (диффузия по графу + обучаемое
линейное взаимодействие признаков, см. `ode_solvers.ode_solver_wcgnn`) ->
линейный классификатор.
"""

from encoders import *
from ode_solvers.ode_solver_wcgnn import *


class WGNN(nn.Module):
    """WCGNN: encoder -> ODEblockW (диффузия + взаимодействие признаков) -> классификатор."""

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
        self.T = time

        self.encoder = self._get_encoder()
        # скрытая размерность ODE-блока в 2 раза больше hidden_dim -- см.
        # комментарий в forward про вспомогательный нулевой блок c_aux
        self.odeblock = ODEblockW(
            ODEFuncW(2 * opt['hidden_dim'], 2 * opt['hidden_dim'], opt, adj, deg),
            t=torch.tensor([0, self.T])
        )
        self.classifier = nn.Linear(opt['hidden_dim'], opt['num_class'])

        if opt['cpu']:
            self.cpu()

    def _get_encoder(self):
        """Создаёт энкодер входных признаков по типу из `opt['encoder_type']`.

        Raises:
            ValueError: если указан неизвестный `encoder_type`.
        """
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
        """Сбрасывает энкодер и классификатор к начальному состоянию."""
        self.encoder = nn.Linear(self.opt['num_feature'], self.opt['hidden_dim'])
        self.classifier.reset_parameters()

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

        return self.classifier(z)
