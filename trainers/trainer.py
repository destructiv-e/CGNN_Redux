"""Цикл обучения и оценки модели (CGNN/WCGNN).

Содержит:
    get_optimizer -- фабрику оптимизаторов torch по имени.
    Meter         -- простой счётчик/усреднитель числовых значений
                     (используется для статистики числа вызовов ODE-функции).
    Trainer       -- собственно обучение (update/updatew), оценку (evaluate),
                     инференс (predict) и сохранение/загрузку чекпоинтов.
"""

import math
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim import Optimizer


def get_optimizer(name, parameters, lr, weight_decay=0):
    """Создаёт оптимизатор torch по короткому имени.

    Args:
        name: одно из 'sgd', 'rmsprop', 'adagrad', 'adam', 'adamax'.
        parameters: обучаемые параметры модели.
        lr: learning rate.
        weight_decay: коэффициент L2-регуляризации.

    Returns:
        torch.optim.Optimizer: созданный оптимизатор.

    Raises:
        Exception: если `name` не поддерживается.
    """
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'rmsprop':
        return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adagrad':
        return torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adam':
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adamax':
        return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))


# Counter of forward and backward passes.
class Meter(object):
    """Счётчик числовых значений с накоплением среднего.

    Используется, чтобы отслеживать число вызовов правой части ODE (nfe)
    на прямом и обратном проходе -- отражает "эффективную глубину" сети.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Сбрасывает накопленную статистику."""
        self.val = None
        self.sum = 0
        self.cnt = 0

    def update(self, val):
        """Добавляет новое значение в статистику."""
        self.val = val
        self.sum += val
        self.cnt += 1

    def get_average(self):
        """Возвращает среднее по всем добавленным значениям (0, если их не было)."""
        if self.cnt == 0:
            return 0
        return self.sum / self.cnt

    def get_value(self):
        """Возвращает последнее добавленное значение."""
        return self.val


class Trainer(object):
    """Обучение, оценка и инференс модели (CGNN/WCGNN)."""

    def __init__(self, opt, model):
        """
        Args:
            opt: словарь гиперпараметров (нужны 'cuda', 'optimizer', 'lr', 'decay').
            model: модель (`models.gnn.GNN` или `models.wgnn.WGNN`).
        """
        self.opt = opt
        self.model = model
        self.fm = Meter()  # число вызовов ODE-функции на forward-проходе
        self.bm = Meter()  # число вызовов ODE-функции на backward-проходе
        self.criterion = nn.CrossEntropyLoss()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:
            self.criterion.cuda()
        self.optimizer = get_optimizer(self.opt['optimizer'], self.parameters, self.opt['lr'], self.opt['decay'])

    def reset(self):
        """Сбрасывает модель и создаёт новый оптимизатор с теми же гиперпараметрами."""
        self.model.reset()
        self.optimizer = get_optimizer(self.opt['optimizer'], self.parameters, self.opt['lr'], self.opt['decay'])

    # Train model with hard labels.
    def update(self, inputs, target, idx):
        """Один шаг обучения CGNN (`GNN`) на объектах с индексами `idx`.

        Args:
            inputs: признаки всех узлов.
            target: истинные метки всех узлов.
            idx: индексы узлов обучающей выборки.

        Returns:
            float: значение функции потерь на этом шаге.
        """
        if self.opt['cuda']:
            inputs = inputs.cuda()
            target = target.cuda()
            idx = idx.cuda()

        self.model.train()
        self.optimizer.zero_grad()

        logits = self.model(inputs)
        loss = self.criterion(logits[idx], target[idx])

        self.fm.update(self.model.odeblock.nfe)
        self.model.odeblock.nfe = 0

        loss.backward()
        self.optimizer.step()

        self.bm.update(self.model.odeblock.nfe)
        self.model.odeblock.nfe = 0

        return loss.item()

    def updatew(self, inputs, target, idx):
        """Один шаг обучения WCGNN (`WGNN`) на объектах с индексами `idx`.

        Помимо обычного шага оптимизации, после `optimizer.step()`
        матрица взаимодействия признаков `W` итеративно приближается к
        ортогональной (шаг Бьорка -- Боуи, Bjorck orthogonalization step),
        чтобы диффузия признаков оставалась численно устойчивой.

        Args:
            inputs: признаки всех узлов.
            target: истинные метки всех узлов.
            idx: индексы узлов обучающей выборки.

        Returns:
            float: значение функции потерь на этом шаге.
        """
        if self.opt['cuda']:
            inputs = inputs.cuda()
            target = target.cuda()
            idx = idx.cuda()

        self.model.train()
        self.optimizer.zero_grad()

        logits = self.model(inputs)
        loss = self.criterion(logits[idx], target[idx])

        self.fm.update(self.model.odeblock.odefunc.nfe)
        self.model.odeblock.odefunc.nfe = 0

        loss.backward()
        self.optimizer.step()

        # шаг ортогонализации W: W <- (1 + beta) W - beta W (W^T W)
        W = self.model.odeblock.odefunc.w.data
        beta = 0.5
        W.copy_((1 + beta) * W - beta * W.mm(W.transpose(0, 1).mm(W)))

        self.bm.update(self.model.odeblock.odefunc.nfe)
        self.model.odeblock.odefunc.nfe = 0

        return loss.item()

    # Train model with soft labels, e.g., [0.1, 0.2, 0.7].
    def update_soft(self, inputs, target, idx):
        """Один шаг обучения с мягкими (soft) метками, например [0.1, 0.2, 0.7].

        Не используется текущими скриптами обучения (`trainers.train`),
        оставлен как альтернативный вариант обучения с распределениями
        по классам вместо жёстких меток.

        Args:
            inputs: признаки всех узлов.
            target: распределения по классам для всех узлов.
            idx: индексы узлов обучающей выборки.

        Returns:
            float: значение функции потерь на этом шаге.
        """
        if self.opt['cuda']:
            inputs = inputs.cuda()
            target = target.cuda()
            idx = idx.cuda()

        self.model.train()
        self.optimizer.zero_grad()

        logits = self.model(inputs)
        logits = torch.log_softmax(logits, dim=-1)
        loss = -torch.mean(torch.sum(target[idx] * logits[idx], dim=-1))

        self.fm.update(self.model.odefunc.ncall)
        self.model.odefunc.ncall = 0

        loss.backward()
        self.optimizer.step()

        self.bm.update(self.model.odefunc.ncall)
        self.model.odefunc.ncall = 0

        return loss.item()

    # Evaluate model.
    def evaluate(self, inputs, target, idx):
        """Оценивает модель (без обучения) на объектах с индексами `idx`.

        Args:
            inputs: признаки всех узлов.
            target: истинные метки всех узлов.
            idx: индексы узлов для оценки (dev/test).

        Returns:
            tuple:
                loss (float): значение функции потерь.
                preds (torch.Tensor): предсказанные классы.
                accuracy (float): доля верных предсказаний.
        """
        if self.opt['cuda']:
            inputs = inputs.cuda()
            target = target.cuda()
            idx = idx.cuda()

        self.model.eval()

        logits = self.model(inputs)
        loss = self.criterion(logits[idx], target[idx])
        preds = torch.max(logits[idx], dim=1)[1]
        correct = preds.eq(target[idx]).double()
        accuracy = correct.sum() / idx.size(0)

        return loss.item(), preds, accuracy.item()

    def predict(self, inputs, tau=1):
        """Возвращает вероятности классов для всех узлов (без обучения).

        Args:
            inputs: признаки всех узлов.
            tau: температура softmax (>1 -- более сглаженное распределение).

        Returns:
            torch.Tensor: вероятности классов [num_nodes x num_class].
        """
        if self.opt['cuda']:
            inputs = inputs.cuda()

        self.model.eval()

        logits = self.model(inputs) / tau

        logits = torch.softmax(logits, dim=-1).detach()

        return logits

    def save(self, filename):
        """Сохраняет веса модели и состояние оптимизатора в файл.

        Ошибка при сохранении не прерывает выполнение, а только логируется.
        """
        params = {
                'model': self.model.state_dict(),
                'optim': self.optimizer.state_dict()
                }
        try:
            torch.save(params, filename)
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def load(self, filename):
        """Загружает веса модели и состояние оптимизатора из файла.

        При ошибке загрузки печатает сообщение и завершает процесс.
        """
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optim'])
