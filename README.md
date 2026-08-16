# GNN as Dynamical Systems

Магистерская диссертация, СПбГУ, математико-механический факультет, кафедра прикладной кибернетики.

Репозиторий содержит код и эксперименты по исследованию графовых нейронных сетей (GNN),
рассматриваемых через призму теории динамических систем: сеть задаётся как ODE-блок
(непрерывная диффузия по графу, интегрируемая `torchdiffeq`), а не как стек дискретных слоёв.
В фокусе — анализ устойчивости, сходимости, методов численного интегрирования и
воспроизводимость экспериментов.

Реализованы два варианта модели:

- **CGNN** (`GNN`) — базовая диффузия по графу с обучаемым коэффициентом `alpha` на рёбрах.
- **WCGNN** (`WGNN`) — версия с дополнительной обучаемой матрицей взаимодействия признаков `W`,
  которая на каждом шаге оптимизации проецируется к (почти) ортогональной для устойчивости.

## Структура проекта

```
CGNN_Redux/
├── data_loader.py            # загрузка графа/фич/меток из текстового формата (Vocab, Graph, EntityLabel, EntityFeature)
├── encoders.py                # энкодеры входных признаков (EncoderWithDropout / EncoderWithoutDropout / EncoderMLP)
├── requirements.txt            # зависимости проекта
│
├── models/                     # архитектуры моделей
│   ├── gnn.py                  #   GNN  — обёртка: энкодер -> ODEblock (CGNN) -> классификатор
│   └── wgnn.py                 #   WGNN — обёртка: энкодер -> ODEblockW (WCGNN) -> классификатор
│
├── ode_solvers/                 # правые части ODE и обвязка интегрирования (torchdiffeq)
│   ├── ode_solver_cgnn.py       #   ODEFunc / ODEblock  — диффузия по графу для CGNN
│   └── ode_solver_wcgnn.py      #   ODEFuncW / ODEblockW — диффузия + обучаемое взаимодействие признаков для WCGNN
│
├── trainers/                    # цикл обучения/оценки
│   ├── trainer.py                #   Trainer — update/updatew/evaluate/predict/save/load
│   └── train.py                  #   CLI-точка входа: парсинг аргументов, загрузка данных, запуск обучения
│
├── run_files/                   # готовые конфигурации запуска (гиперпараметры под конкретный датасет/модель)
│   ├── run_cgnn.py               #   запуск CGNN на dataset/out_cora
│   └── run_wcgnn.py              #   запуск WCGNN на dataset/out_cora
│
└── dataset/
    ├── pyg_to_custom_format.py   # скачивает датасеты через PyTorch Geometric и конвертирует их в текстовый формат ниже
    ├── pyg_data/                 # кэш "сырых" датасетов PyTorch Geometric (Planetoid, WebKB, Actor, ...)
    └── out_<name>/                # готовые датасеты в текстовом формате: net.txt, feature.txt, label.txt, train/dev/test.txt
```

Каждый пакет (`models`, `ode_solvers`, `trainers`) оформлен как обычный Python-пакет
(есть `__init__.py`), поэтому импорты между модулями абсолютные, например
`from models.gnn import GNN` или `from trainers.trainer import Trainer`.

### Формат датасета (`dataset/out_<name>/`)

| Файл | Формат строки | Описание |
|---|---|---|
| `net.txt` | `src_id \t dst_id \t weight` | рёбра графа |
| `feature.txt` | `node_id \t feat_idx:value feat_idx:value ...` | разреженные признаки узла |
| `label.txt` | `node_id \t label` | метки классов (только размеченные узлы) |
| `train.txt` / `dev.txt` / `test.txt` | `node_id` (по одному на строку) | сплиты по индексам узлов |

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Требуется Python 3.9+. Проект работает как на CPU, так и на GPU — устройство выбирается
автоматически (`--cuda` включён по умолчанию, но при отсутствии CUDA обучение идёт на CPU).

## Быстрый старт

Готовые конфигурации с гиперпараметрами лежат в `run_files/` и запускаются из корня проекта:

```bash
python run_files/run_cgnn.py     # CGNN на dataset/out_cora
python run_files/run_wcgnn.py    # WCGNN на dataset/out_cora
```

Внутри они вызывают `python -m trainers.train` с нужными аргументами (это единая точка входа
для обучения — гиперпараметры для конкретного эксперимента задаются либо в `run_files/*.py`,
либо напрямую через CLI):

```bash
python -m trainers.train --dataset dataset/out_cora --epoch 200 --hidden_dim 16 \
    --lr 0.0047 --decay 5e-4 --self_link_weight 0.555 --alpha 0.918 --time 12.1
```

Для WCGNN добавьте `--weight True`.

### Основные аргументы `trainers/train.py`

| Аргумент | По умолчанию | Описание |
|---|---|---|
| `--dataset` | `data` | путь к директории датасета (формат см. выше) |
| `--epoch` | `10` | число эпох обучения |
| `--hidden_dim` | `16` | размерность скрытого представления |
| `--lr` | `0.01` | learning rate |
| `--decay` | `5e-4` | weight decay |
| `--optimizer` | `adam` | `sgd` / `rmsprop` / `adagrad` / `adam` / `adamax` |
| `--self_link_weight` | `1.0` | вес self-loop при симметризации графа |
| `--alpha` | `1.0` | начальное значение обучаемого коэффициента диффузии |
| `--time` | `1.0` | конечное время интегрирования ODE |
| `--dropout` / `--input_dropout` | `0.5` | dropout скрытого слоя / входа |
| `--weight` | `False` | `True` — использовать WCGNN (`WGNN`) вместо CGNN (`GNN`) |
| `--cuda` | `True`, если доступна CUDA | использовать GPU |
| `--cpu` | `False` | принудительно использовать CPU |
| `--seed` | `1` | random seed |

## Добавление нового датасета

```bash
cd dataset
python pyg_to_custom_format.py --dataset citeseer --outdir ./out_citeseer
```

Поддерживаемые датасеты: `cora`, `citeseer`, `pubmed`, `texas`, `cornell`, `wisconsin`,
`film`, `chameleon`, `squirrel`, `amazon-computers`, `amazon-photo`, `coauthor-cs`,
`coauthor-physics`, `wikics`, `roman-empire`, `amazon-ratings`, `minesweeper`, `tolokers`,
`questions`. Полный список опций — `python dataset/pyg_to_custom_format.py --help`.

В репозитории уже сконвертированы: `out_cora`, `out_citeseer`, `out_pubmed`, `out_film`, `out_texas`.

## Автор

[Толстокоров Савелий] | СПбГУ, математико-механический факультет
