"""Правая часть ODE и обвязка интегрирования для взвешенной модели WCGNN.

По сравнению с базовой CGNN (`ode_solvers.ode_solver_cgnn`) сюда добавлено
обучаемое линейное взаимодействие признаков через матрицу `w = W diag(d) W^T`
(при обучении `W` шагом проекции в `Trainer.updatew` приближается к
ортогональной, а `d` ограничена в [0, 1] через `torch.clamp`) -- это
позволяет модели дополнительно смешивать и масштабировать сами признаки, а
не только агрегировать их по графу:

    dx/dt = alpha * 0.5 * (A x - x) + x W - x + x0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint


class ODEFuncW(nn.Module):
    """Правая часть ODE: диффузия по графу + обучаемое взаимодействие признаков."""

    def __init__(self, in_features, out_features, opt, adj, deg):
        """
        Args:
            in_features: размерность входного представления (для справки).
            out_features: размерность выходного представления (для справки).
            opt: словарь гиперпараметров модели (нужны 'alpha', 'hidden_dim').
            adj: разреженная матрица смежности графа.
            deg: степени узлов (не используется явно в этой версии).
        """
        super().__init__()
        self.opt = opt
        self.adj = adj
        self.x0 = None  # начальное представление узлов, задаётся через ODEblockW.set_x0
        self.nfe = 0  # счётчик вызовов forward (number of function evaluations)
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = opt['alpha']
        # обучаемый коэффициент диффузии alpha, отдельный на каждое ребро
        self.alpha_train = nn.Parameter(self.alpha * torch.ones(adj.shape[1]))
        # W -- обучаемая матрица взаимодействия признаков, поддерживается
        # (почти) ортогональной шагом в Trainer.updatew
        self.w = nn.Parameter(torch.eye(2 * opt['hidden_dim']))
        # d -- диагональное масштабирование, обрезается в [0, 1] перед использованием
        self.d = nn.Parameter(torch.zeros(2 * opt['hidden_dim']) + 1)

    def forward(self, t, x):
        """Вычисляет dx/dt в момент времени `t` для состояния `x`.

        Args:
            t: текущее время (не используется явно -- уравнение автономное).
            x: состояние узлов [num_nodes x hidden_dim].

        Returns:
            torch.Tensor: производная dx/dt той же формы, что и `x`.
        """
        self.nfe += 1
        alph = F.sigmoid(self.alpha_train).unsqueeze(dim=1)  # alpha в (0, 1) на каждом ребре
        ax = torch.spmm(self.adj, x)  # агрегация признаков соседей: A @ x
        d = torch.clamp(self.d, min=0, max=1)
        w = torch.mm(self.w * d, torch.t(self.w))  # симметричная матрица взаимодействия признаков
        xw = torch.spmm(x, w)  # применение взаимодействия признаков: x @ w
        return alph * 0.5 * (ax - x) + xw - x + self.x0


class ODEblockW(nn.Module):
    """Обёртка над `ODEFuncW`, решающая ODE на интервале времени `t`."""

    def __init__(self, odefunc, t=torch.tensor([0, 1])):
        """
        Args:
            odefunc: правая часть ODE (`ODEFuncW`).
            t: тензор из двух элементов [t_start, t_end] -- интервал интегрирования.
        """
        super().__init__()
        self.t = t
        self.odefunc = odefunc
        self.nfe = 0

    def set_x0(self, x0):
        """Задаёт начальное состояние x0 для правой части ODE (без градиента)."""
        self.odefunc.x0 = x0.clone().detach()

    def forward(self, x):
        """Интегрирует ODE от `x` на интервале `self.t` и возвращает x(t_end).

        Args:
            x: начальное состояние узлов [num_nodes x hidden_dim].

        Returns:
            torch.Tensor: состояние узлов в момент t_end.
        """
        self.nfe += 1
        t = self.t.type_as(x)
        return odeint(self.odefunc, x, t)[1]

    def __repr__(self):
        return f"{self.__class__.__name__}(Time Interval {self.t[0].item()} -> {self.t[1].item()})"
