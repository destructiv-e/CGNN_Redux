"""Энкодеры входных признаков узлов.

Каждый энкодер отображает разреженный/плотный вектор признаков узла
(размерности `input_dim`, обычно one-hot по словам/атрибутам) в скрытое
представление размерности `hidden_dim`, с которым дальше работает ODE-блок
модели (`models.gnn.GNN` / `models.wgnn.WGNN`).
"""

import torch.nn as nn
import torch.nn.functional as F


class EncoderWithDropout(nn.Module):
    """Линейный энкодер с dropout на входе (используется по умолчанию)."""

    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class EncoderWithoutDropout(nn.Module):
    """Линейный энкодер без dropout."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        return self.linear(x)


class EncoderMLP(nn.Module):
    """Двухслойный MLP-энкодер с ReLU в промежуточном слое."""

    def __init__(self, input_dim, hidden_dim, intermediate_dim=128):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, intermediate_dim)
        self.linear2 = nn.Linear(intermediate_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        return self.linear2(x)
