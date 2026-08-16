"""Правая часть ODE и обвязка интегрирования для базовой модели CGNN.

Диффузия по графу задаётся непрерывным во времени процессом

    dx/dt = alpha * 0.5 * (A x - x) + x0,

где `A` -- (симметрично нормализованная) матрица смежности, `alpha` --
обучаемый на каждом ребре коэффициент диффузии, а `x0` -- представление
узла на входе в ODE-блок (действует как исходный/утекающий член и
удерживает решение рядом с начальными признаками). Решение x(T) в момент
`T = opt['time']` получается интегрированием этого уравнения с помощью
`torchdiffeq.odeint`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint


class ODEFunc(nn.Module):
    """Правая часть ODE: dx/dt = f(t, x) для диффузии по графу."""

    def __init__(self, in_features, out_features, opt, adj, deg):
        """
        Args:
            in_features: размерность входного представления (не используется
                напрямую, хранится для справки/совместимости интерфейса).
            out_features: размерность выходного представления (аналогично).
            opt: словарь гиперпараметров модели (нужны 'alpha', 'hidden_dim').
            adj: разреженная матрица смежности графа.
            deg: степени узлов (не используется в этом варианте функции,
                хранится для совместимости с WCGNN-версией).
        """
        super().__init__()
        self.opt = opt
        self.adj = adj
        self.x0 = None  # начальное представление узлов, задаётся через ODEblock.set_x0
        self.nfe = 0  # счётчик вызовов forward (number of function evaluations)
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = opt['alpha']
        # обучаемый коэффициент диффузии alpha, отдельный на каждое ребро
        self.alpha_train = nn.Parameter(self.alpha * torch.ones(adj.shape[1]))
        self.w = nn.Parameter(torch.eye(opt['hidden_dim']))
        self.d = nn.Parameter(torch.zeros(opt['hidden_dim']) + 1)

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
        return alph * 0.5 * (ax - x) + self.x0


class ODEblock(nn.Module):
    """Обёртка над `ODEFunc`, решающая ODE на интервале времени `t`."""

    def __init__(self, odefunc, t=torch.tensor([0, 1])):
        """
        Args:
            odefunc: правая часть ODE (`ODEFunc`).
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
        time_range = f"{self.t[0].item()} -> {self.t[1].item()}"
        return f"{self.__class__.__name__}(Time Interval {time_range})"
