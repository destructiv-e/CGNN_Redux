"""Загрузка графовых датасетов из собственного текстового формата.

Формат описан в `dataset/pyg_to_custom_format.py` и в README проекта:
    net.txt      src_id \t dst_id \t weight
    feature.txt  node_id \t feat_idx:value feat_idx:value ...
    label.txt    node_id \t label
    train/dev/test.txt   node_id (по одному на строку)

Модуль предоставляет четыре класса:
    Vocab         -- словарь "строка <-> индекс", строится по одной или
                     нескольким колонкам TSV-файла.
    EntityLabel   -- отображение "индекс узла -> индекс метки класса".
    EntityFeature -- отображение "индекс узла -> разреженные признаки",
                     с конвертацией в плотное (one-hot/multi-hot) представление.
    Graph         -- список рёбер графа и операции над ним (симметризация,
                     нормализация весов, построение матрицы смежности).
"""

import sys
import os
import math
import numpy as np
import torch
from torch.autograd import Variable


class Vocab(object):
    """Словарь "строка -> индекс", построенный по колонкам TSV-файла.

    Каждая ячейка указанных колонок может содержать несколько
    пробел-разделённых токенов вида "token" или "token:value" (второй
    формат используется для признаков) -- в словарь попадает только
    часть до ":".
    """

    def __init__(self, file_name, cols, with_padding=False):
        """
        Args:
            file_name: путь к TSV-файлу.
            cols: индексы колонок, из которых нужно собирать токены.
            with_padding: добавить служебный токен "<pad>" с индексом 0.
        """
        self.itos = []
        self.stoi = {}
        self.vocab_size = 0

        if with_padding:
            string = '<pad>'
            self.stoi[string] = self.vocab_size
            self.itos.append(string)
            self.vocab_size += 1

        fi = open(file_name, 'r')
        for line in fi:
            line = line.strip()
            if not line:  # пропускаем пустые строки
                continue
            items = line.split('\t')
            # если колонок меньше, чем нам нужно, то пропускаем
            if len(items) <= max(cols):
                continue
            for col in cols:
                item = items[col]
                # в случае, если признаков нет, item может быть пустой строкой, тогда пропускаем
                if not item:
                    continue
                strings = item.strip().split(' ')
                for string in strings:
                    string = string.split(':')[0]
                    if string not in self.stoi:
                        self.stoi[string] = self.vocab_size
                        self.itos.append(string)
                        self.vocab_size += 1
        fi.close()

    def __len__(self):
        return self.vocab_size


class EntityLabel(object):
    """Отображение "индекс узла -> индекс метки класса".

    Строится по `label.txt`; узлам без метки соответствует -1.
    """

    def __init__(self, file_name, entity, label):
        """
        Args:
            file_name: путь к файлу с метками (`label.txt`).
            entity: пара (Vocab узлов, индекс колонки с id узла).
            label: пара (Vocab меток, индекс колонки с меткой).
        """
        self.vocab_n, self.col_n = entity
        self.vocab_l, self.col_l = label
        self.itol = [-1 for k in range(self.vocab_n.vocab_size)]

        fi = open(file_name, 'r')
        for line in fi:
            line = line.strip()
            if not line:
                continue
            items = line.split('\t')
            if len(items) <= max(self.col_n, self.col_l):
                continue
            sn, sl = items[self.col_n], items[self.col_l]
            n = self.vocab_n.stoi.get(sn, -1)
            l = self.vocab_l.stoi.get(sl, -1)
            if n == -1:
                continue
            self.itol[n] = l
        fi.close()


class EntityFeature(object):
    """Разреженные признаки узлов и их конвертация в плотный вид.

    После загрузки `itof[node]` -- список пар (feature_id, weight).
    Методы `to_one_hot`/`to_index` строят из этого разреженного
    представления плотные тензоры, пригодные для подачи в модель.
    """

    def __init__(self, file_name, entity, feature):
        """
        Args:
            file_name: путь к файлу признаков (`feature.txt`).
            entity: пара (Vocab узлов, индекс колонки с id узла).
            feature: пара (Vocab признаков, индекс колонки с признаками).
        """
        self.vocab_n, self.col_n = entity
        self.vocab_f, self.col_f = feature
        self.itof = [[] for k in range(len(self.vocab_n))]
        self.one_hot = []

        fi = open(file_name, 'r')
        for line in fi:
            line = line.strip()
            if not line:
                continue
            items = line.split('\t')
            if len(items) <= max(self.col_n, self.col_f):
                continue
            sn, sf = items[self.col_n], items[self.col_f]
            n = self.vocab_n.stoi.get(sn, -1)
            if n == -1:
                continue
            # если sf пустая строка, то пропускаем
            if not sf:
                continue
            for s in sf.strip().split(' '):
                f = self.vocab_f.stoi.get(s.split(':')[0], -1)
                w = float(s.split(':')[1])
                if f == -1:
                    continue
                self.itof[n].append((f, w))
        fi.close()

    def to_one_hot(self, binary=False):
        """Строит плотную матрицу признаков `self.one_hot` [num_nodes x num_features].

        Веса признаков каждого узла нормируются на их сумму (получаются
        доли), либо, если `binary=True`, сначала заменяются на 1.0 --
        тогда после нормировки получается обычный one-hot/multi-hot вектор.

        Args:
            binary: игнорировать исходные веса признаков, считать их
                бинарными (наличие/отсутствие).
        """
        self.one_hot = [[0 for j in range(len(self.vocab_f))] for i in range(len(self.vocab_n))]
        for k in range(len(self.vocab_n)):
            sm = 0
            for fid, wt in self.itof[k]:
                if binary:
                    wt = 1.0
                sm += wt
            for fid, wt in self.itof[k]:
                if binary:
                    wt = 1.0
                if sm > 0:
                    self.one_hot[k][fid] = wt / sm
                else:
                    self.one_hot[k][fid] = 0

    def to_index(self):
        """Строит `self.index` [num_nodes x max_features] -- индексы признаков
        каждого узла, дополненные нулями до одинаковой длины (padding).
        """
        max_length = max([len(fs) for fs in self.itof]) if self.itof else 0
        self.index = [[int(0) for j in range(max_length)] for i in range(len(self.vocab_n))]
        for k in range(len(self.vocab_n)):
            for i, (fid, wt) in enumerate(self.itof[k]):
                self.index[k][i] = int(fid)


class Graph(object):
    """Список рёбер графа и операции над ним: симметризация, нормализация
    весов, построение разреженной/плотной матрицы смежности.
    """

    def __init__(self, file_name, entity, weight=None):
        """
        Args:
            file_name: путь к файлу с рёбрами (`net.txt`).
            entity: тройка (Vocab узлов, индекс колонки src, индекс колонки dst).
            weight: индекс колонки с весом ребра, либо None (тогда вес = 1).
        """
        self.vocab_n, self.col_u, self.col_v = entity
        self.col_w = weight
        self.edges = []

        self.node_size = -1

        self.eid2iid = None
        self.iid2eid = None

        self.adj_w = None
        self.adj_t = None

        with open(file_name, 'r') as fi:
            for line in fi:
                line = line.strip()
                if not line:
                    continue
                items = line.split('\t')
                if len(items) <= max(self.col_u, self.col_v):
                    continue
                su, sv = items[self.col_u], items[self.col_v]
                sw = items[self.col_w] if self.col_w != None else None

                u, v = self.vocab_n.stoi.get(su, -1), self.vocab_n.stoi.get(sv, -1)
                w = float(sw) if sw != None else 1

                if u == -1 or v == -1 or w <= 0:
                    continue

                self.edges += [(u, v, w)]

    def get_node_size(self):
        return self.node_size

    def get_edge_size(self):
        return len(self.edges)

    def to_symmetric(self, self_link_weight=1.0):
        """Симметризует граф и нормализует веса рёбер.

        Для каждой пары узлов берётся ребро с большим весом в обе стороны
        (или единственное ребро, если веса в обеих направлениях равны),
        затем при `self_link_weight > 0` добавляются петли (self-loops) с
        этим весом. После этого веса нормализуются симметрично по степеням
        узлов: w(u, v) -> w(u, v) / sqrt(deg(u) * deg(v)) -- аналог
        нормализации D^-1/2 A D^-1/2, как в GCN.

        Args:
            self_link_weight: вес добавляемой петли узла на самого себя;
                при <= 0 петли не добавляются.

        Returns:
            dict: невзвешенная степень каждого узла (сумма весов исходящих
            рёбер до нормализации) -- используется как `deg` при обучении.
        """
        vocab = set()
        for u, v, w in self.edges:
            vocab.add(u)
            vocab.add(v)

        pair2wt = dict()
        for u, v, w in self.edges:
            pair2wt[(u, v)] = w

        edges_ = list()
        for (u, v), w in pair2wt.items():
            if u == v:
                continue
            w_ = pair2wt.get((v, u), -1)
            if w > w_:
                edges_ += [(u, v, w), (v, u, w)]
            elif w == w_:
                edges_ += [(u, v, w)]
        if self_link_weight > 0:
            for k in vocab:
                edges_ += [(k, k, self_link_weight)]

        d = dict()
        for u, v, w in edges_:
            d[u] = d.get(u, 0.0) + w

        self.edges = [(u, v, w / math.sqrt(d[u] * d[v])) for u, v, w in edges_]
        return d

    def get_sparse_adjacency(self, cuda=True):
        """Строит разреженную матрицу смежности [num_nodes x num_nodes].

        Args:
            cuda: разместить тензор на GPU.

        Returns:
            torch.sparse.FloatTensor: разреженная матрица смежности.
        """
        shape = torch.Size([self.vocab_n.vocab_size, self.vocab_n.vocab_size])

        us, vs, ws = [], [], []
        for u, v, w in self.edges:
            us += [u]
            vs += [v]
            ws += [w]
        index = torch.LongTensor([us, vs])
        value = torch.Tensor(ws)
        if cuda:
            index = index.cuda()
            value = value.cuda()
        adj = torch.sparse.FloatTensor(index, value, shape)
        if cuda:
            adj = adj.cuda()

        return adj

    def get_dense_adjenccy(self, cuda=True):
        """Строит плотную матрицу смежности [num_nodes x num_nodes].

        Args:
            cuda: разместить тензор на GPU.

        Returns:
            torch.Tensor: плотная матрица смежности.
        """

        shape = torch.Size([self.vocab_n.vocab_size, self.vocab_n.vocab_size])

        adj = torch.zeros(shape, dtype=torch.float)
        for u, v, w in self.edges:
            adj[u, v] = w

        if cuda:
            adj = adj.cuda()

        return adj
