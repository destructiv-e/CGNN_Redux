#!/usr/bin/env python3
"""
pyg_to_custom_format.py

Скачивает граф-датасеты для задачи классификации узлов через
PyTorch Geometric (torch_geometric.datasets) и конвертирует их в
собственный текстовый формат:

    feature.txt   node_id \t feat_idx:value feat_idx:value ...   (только ненулевые, отсортировано по feat_idx)
    net.txt       src_id \t dst_id \t 1                          (неориентированный граф, обе стороны, без петель/дублей, отсортировано)
    label.txt     node_id \t label                               (только train+dev+test узлы, сначала train, потом dev, потом test)
    train.txt     node_id                                        (по одному на строку)
    dev.txt       node_id
    test.txt      node_id

Поддерживаемые датасеты (см. DATASET_REGISTRY):
    cora, citeseer, pubmed                         -> Planetoid
    texas, cornell, wisconsin                       -> WebKB
    film (он же actor)                               -> Actor
    chameleon, squirrel                              -> WikipediaNetwork
    amazon-computers, amazon-photo                   -> Amazon
    coauthor-cs, coauthor-physics                    -> Coauthor
    wikics                                           -> WikiCS
    roman-empire, amazon-ratings, minesweeper,
    tolokers, questions                              -> HeterophilousGraphDataset

Примеры запуска:
    python pyg_to_custom_format.py --dataset texas --outdir ./out_texas
    python pyg_to_custom_format.py --dataset citeseer --outdir ./out_citeseer
    python pyg_to_custom_format.py --dataset film --outdir ./out_film --split-idx 3
    python pyg_to_custom_format.py --dataset chameleon --outdir ./out_chameleon
    python pyg_to_custom_format.py --dataset cora --outdir ./out_cora
    python pyg_to_custom_format.py --dataset pubmed --outdir ./out_pubmed
    python pyg_to_custom_format.py --dataset wisconsin --outdir ./out_wisconsin --split-idx 2
    python pyg_to_custom_format.py --dataset squirrel --outdir ./out_squirrel --geom-gcn-preprocess False
    python pyg_to_custom_format.py --dataset amazon-computers --outdir ./out_amazon_computers
    python pyg_to_custom_format.py --dataset amazon-photo --outdir ./out_amazon_photo
    python pyg_to_custom_format.py --dataset coauthor-cs --outdir ./out_coauthor_cs
    python pyg_to_custom_format.py --dataset coauthor-physics --outdir ./out_coauthor_physics
    python pyg_to_custom_format.py --dataset wikics --outdir ./out_wikics
    python pyg_to_custom_format.py --dataset roman-empire --outdir ./out_roman_empire
    python pyg_to_custom_format.py --dataset amazon-ratings --outdir ./out_amazon_ratings
    python pyg_to_custom_format.py --dataset minesweeper --outdir ./out_minesweeper
    python pyg_to_custom_format.py --dataset tolokers --outdir ./out_tolokers
    python pyg_to_custom_format.py --dataset questions --outdir ./out_questions

Требуется: pip install torch torch_geometric
"""

import argparse
import os
import random
import sys
from collections import defaultdict


# ---------------------------------------------------------------------------
# DATASET_REGISTRY: имя -> функция () -> torch_geometric.data.Data
# ---------------------------------------------------------------------------

def _load_planetoid(name, root):
    from torch_geometric.datasets import Planetoid
    ds = Planetoid(root=os.path.join(root, "Planetoid"), name=name)
    return ds[0]


def _load_webkb(name, root):
    from torch_geometric.datasets import WebKB
    ds = WebKB(root=os.path.join(root, "WebKB"), name=name)
    return ds[0]


def _load_actor(root):
    from torch_geometric.datasets import Actor
    ds = Actor(root=os.path.join(root, "Actor"))
    return ds[0]


def _load_wikipedia_network(name, root, geom_gcn_preprocess):
    from torch_geometric.datasets import WikipediaNetwork
    ds = WikipediaNetwork(
        root=os.path.join(root, "WikipediaNetwork"),
        name=name,
        geom_gcn_preprocess=geom_gcn_preprocess,
    )
    return ds[0]


def _load_amazon(name, root):
    from torch_geometric.datasets import Amazon
    ds = Amazon(root=os.path.join(root, "Amazon"), name=name)
    return ds[0]


def _load_coauthor(name, root):
    from torch_geometric.datasets import Coauthor
    ds = Coauthor(root=os.path.join(root, "Coauthor"), name=name)
    return ds[0]


def _load_wikics(root):
    from torch_geometric.datasets import WikiCS
    ds = WikiCS(root=os.path.join(root, "WikiCS"))
    return ds[0]


def _load_heterophilous(name, root):
    from torch_geometric.datasets import HeterophilousGraphDataset
    ds = HeterophilousGraphDataset(root=os.path.join(root, "HeterophilousGraphDataset"), name=name)
    return ds[0]


def build_registry(root, geom_gcn_preprocess):
    """Строит DATASET_REGISTRY: dataset_name (lowercase) -> callable () -> Data."""
    return {
        "cora": lambda: _load_planetoid("Cora", root),
        "citeseer": lambda: _load_planetoid("CiteSeer", root),
        "pubmed": lambda: _load_planetoid("PubMed", root),

        "texas": lambda: _load_webkb("Texas", root),
        "cornell": lambda: _load_webkb("Cornell", root),
        "wisconsin": lambda: _load_webkb("Wisconsin", root),

        "film": lambda: _load_actor(root),
        "actor": lambda: _load_actor(root),

        "chameleon": lambda: _load_wikipedia_network("chameleon", root, geom_gcn_preprocess),
        "squirrel": lambda: _load_wikipedia_network("squirrel", root, geom_gcn_preprocess),

        "amazon-computers": lambda: _load_amazon("Computers", root),
        "amazon-photo": lambda: _load_amazon("Photo", root),

        "coauthor-cs": lambda: _load_coauthor("CS", root),
        "coauthor-physics": lambda: _load_coauthor("Physics", root),

        "wikics": lambda: _load_wikics(root),

        "roman-empire": lambda: _load_heterophilous("Roman-empire", root),
        "amazon-ratings": lambda: _load_heterophilous("Amazon-ratings", root),
        "minesweeper": lambda: _load_heterophilous("Minesweeper", root),
        "tolokers": lambda: _load_heterophilous("Tolokers", root),
        "questions": lambda: _load_heterophilous("Questions", root),
    }


# ---------------------------------------------------------------------------
# Разбиение на train/dev/test
# ---------------------------------------------------------------------------

def has_official_splits(data):
    """Есть ли готовые train_mask/val_mask/test_mask у объекта Data."""
    return hasattr(data, "train_mask") and data.train_mask is not None \
        and hasattr(data, "val_mask") and data.val_mask is not None \
        and hasattr(data, "test_mask") and data.test_mask is not None


def extract_official_split(data, split_idx):
    """Извлекает train/dev/test id-шники из масок Data (поддерживает как 1D, так и 2D [N, num_splits] маски)."""
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask

    if train_mask.dim() == 2:
        num_splits = train_mask.size(1)
        if split_idx >= num_splits:
            raise ValueError(
                f"--split-idx={split_idx} вне диапазона: у датасета доступно {num_splits} сплитов (0..{num_splits - 1})"
            )
        train_ids = train_mask[:, split_idx].nonzero(as_tuple=True)[0].tolist()
        dev_ids = val_mask[:, split_idx].nonzero(as_tuple=True)[0].tolist()
        test_ids = test_mask[:, split_idx].nonzero(as_tuple=True)[0].tolist()
    else:
        if split_idx != 0:
            print(f"  warning: у датасета только один сплит, --split-idx={split_idx} игнорируется")
        train_ids = train_mask.nonzero(as_tuple=True)[0].tolist()
        dev_ids = val_mask.nonzero(as_tuple=True)[0].tolist()
        test_ids = test_mask.nonzero(as_tuple=True)[0].tolist()

    return sorted(train_ids), sorted(dev_ids), sorted(test_ids)


def build_random_split(labels_by_node, train_per_class, dev_size, dev_frac,
                        test_size, test_frac, seed):
    """Стратифицированный train (N на класс, с уменьшением при нехватке узлов) +
    случайный dev/test из остатка (либо фиксированные размеры, либо доли)."""
    rng = random.Random(seed)

    by_class = defaultdict(list)
    for node_idx, lbl in labels_by_node.items():
        by_class[lbl].append(node_idx)
    for lbl in by_class:
        rng.shuffle(by_class[lbl])

    train_ids = []
    used = set()
    for lbl in sorted(by_class):
        class_nodes = by_class[lbl]
        n_avail = len(class_nodes)
        n_take = train_per_class
        if n_avail < train_per_class:
            n_take = max(1, int(round(0.6 * n_avail)))
            print(
                f"  warning: класс {lbl} содержит только {n_avail} узлов (< {train_per_class}), "
                f"берём {n_take} узлов (60%) в train вместо {train_per_class}"
            )
        chosen = class_nodes[:n_take]
        train_ids.extend(chosen)
        used.update(chosen)

    remaining = [n for n in labels_by_node if n not in used]
    rng.shuffle(remaining)

    n_remaining = len(remaining)

    if dev_size is not None:
        n_dev = dev_size
    else:
        n_dev = int(round(dev_frac * n_remaining))

    if test_size is not None:
        n_test = test_size
    else:
        n_test = int(round(test_frac * n_remaining))

    if n_dev + n_test > n_remaining:
        print(
            f"  warning: запрошено dev={n_dev} + test={n_test} > остаток {n_remaining}, "
            f"обрезаем пропорционально"
        )
        total_req = n_dev + n_test
        n_dev = int(n_remaining * (n_dev / total_req))
        n_test = n_remaining - n_dev

    dev_ids = remaining[:n_dev]
    test_ids = remaining[n_dev:n_dev + n_test]

    return sorted(train_ids), sorted(dev_ids), sorted(test_ids)


# ---------------------------------------------------------------------------
# Конвертация
# ---------------------------------------------------------------------------

def convert(data, args):
    used_official_split = False

    # ---- labels ----
    if not hasattr(data, "y") or data.y is None:
        raise ValueError("у датасета отсутствует data.y (метки классов) — конвертация невозможна")

    y = data.y
    if y.dim() > 1:
        # некоторые датасеты (напр. multi-label) хранят y как [N, C]; берём argmax как класс
        print("  warning: data.y имеет более одного столбца, беру argmax по последней оси как метку класса")
        y = y.argmax(dim=-1)

    raw_labels = y.tolist()
    n_nodes = len(raw_labels)

    unique_labels = sorted(set(raw_labels))
    label_remap = {old: new for new, old in enumerate(unique_labels)}
    n_classes = len(unique_labels)
    labels_by_node = {i: label_remap[raw_labels[i]] for i in range(n_nodes)}

    # ---- features ----
    if hasattr(data, "x") and data.x is not None:
        x = data.x
        n_feats = x.size(1)
        has_features = True
    else:
        print("  warning: у датасета нет data.x — feature.txt будет содержать узлы без признаков")
        x = None
        n_feats = 0
        has_features = False

    # ---- split ----
    if has_official_splits(data):
        train_ids, dev_ids, test_ids = extract_official_split(data, args.split_idx)
        used_official_split = True
    else:
        print("  ВНИМАНИЕ: у датасета нет официальных PyG-сплитов train/val/test — "
              "генерирую собственный сплит (НЕ официальный, воспроизводимость только через --seed)")
        train_ids, dev_ids, test_ids = build_random_split(
            labels_by_node,
            args.train_per_class,
            args.dev_size,
            args.dev_frac,
            args.test_size,
            args.test_frac,
            args.seed,
        )

    # ---- edges ----
    edge_index = data.edge_index
    src_list = edge_index[0].tolist()
    dst_list = edge_index[1].tolist()

    edges = set()
    for u, v in zip(src_list, dst_list):
        if u == v:
            continue
        edges.add((u, v))
        edges.add((v, u))

    # ---- write files ----
    os.makedirs(args.outdir, exist_ok=True)

    feat_path = os.path.join(args.outdir, "feature.txt")
    with open(feat_path, "w", encoding="utf-8") as f:
        for node_id in range(n_nodes):
            if has_features:
                row = x[node_id]
                nz = row.nonzero(as_tuple=True)[0].tolist()
                parts = [f"{fi}:{args.float_fmt.format(float(row[fi]))}" for fi in sorted(nz)]
            else:
                parts = []
            f.write(f"{node_id}\t" + " ".join(parts) + "\n")

    net_path = os.path.join(args.outdir, "net.txt")
    with open(net_path, "w", encoding="utf-8") as f:
        for u, v in sorted(edges):
            f.write(f"{u}\t{v}\t1\n")

    label_path = os.path.join(args.outdir, "label.txt")
    all_split_ids = train_ids + dev_ids + test_ids
    with open(label_path, "w", encoding="utf-8") as f:
        for idx in all_split_ids:
            f.write(f"{idx}\t{labels_by_node[idx]}\n")

    def write_ids(path, ids):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(str(i) for i in ids) + "\n")

    write_ids(os.path.join(args.outdir, "train.txt"), train_ids)
    write_ids(os.path.join(args.outdir, "dev.txt"), dev_ids)
    write_ids(os.path.join(args.outdir, "test.txt"), test_ids)

    return {
        "n_nodes": n_nodes,
        "n_feats": n_feats,
        "n_classes": n_classes,
        "n_edges": len(edges),
        "n_train": len(train_ids),
        "n_dev": len(dev_ids),
        "n_test": len(test_ids),
        "used_official_split": used_official_split,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"boolean value expected, got: {v}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dataset", required=True,
                    help="имя датасета (регистр не важен), см. список в описании скрипта")
    ap.add_argument("--outdir", required=True, help="директория для 6 выходных файлов")
    ap.add_argument("--root", default="./pyg_data",
                    help="куда PyG кладёт скачанные сырые файлы (по умолчанию ./pyg_data)")
    ap.add_argument("--seed", type=int, default=42, help="seed для генерации сплита (если нет официального)")
    ap.add_argument("--split-idx", type=int, default=0,
                    help="индекс официального сплита, если их несколько (0..9), по умолчанию 0")
    ap.add_argument("--train-per-class", type=int, default=20,
                    help="сколько узлов на класс брать в train при генерации сплита")
    ap.add_argument("--dev-size", type=int, default=None,
                    help="фиксированный размер dev при генерации сплита (приоритет над --dev-frac)")
    ap.add_argument("--test-size", type=int, default=None,
                    help="фиксированный размер test при генерации сплита (приоритет над --test-frac)")
    ap.add_argument("--dev-frac", type=float, default=0.2,
                    help="доля остатка под dev при генерации сплита, если --dev-size не задан")
    ap.add_argument("--test-frac", type=float, default=0.2,
                    help="доля остатка под test при генерации сплита, если --test-size не задан")
    ap.add_argument("--geom-gcn-preprocess", type=str2bool, default=True,
                    help="для chameleon/squirrel: использовать geom_gcn_preprocess=True (вариант из статьи Geom-GCN), по умолчанию True")
    ap.add_argument("--float-fmt", default="{:.6g}", help="формат вывода значений признаков")
    args = ap.parse_args()

    dataset_key = args.dataset.strip().lower()

    try:
        registry = build_registry(args.root, args.geom_gcn_preprocess)
    except ImportError as e:
        print(f"ошибка импорта torch_geometric: {e}", file=sys.stderr)
        print("установите зависимости: pip install torch torch_geometric", file=sys.stderr)
        sys.exit(1)

    if dataset_key not in registry:
        print(f"ошибка: неизвестный датасет '{args.dataset}'", file=sys.stderr)
        print("доступные датасеты:", file=sys.stderr)
        for name in sorted(registry):
            print(f"  - {name}", file=sys.stderr)
        sys.exit(1)

    print(f"Загружаю датасет '{dataset_key}' (root={args.root})...")
    try:
        data = registry[dataset_key]()
    except Exception as e:
        print(f"ошибка при скачивании/загрузке датасета '{dataset_key}': {e}", file=sys.stderr)
        raise

    print("Конвертирую в целевой формат...")
    try:
        stats = convert(data, args)
    except Exception as e:
        print(f"ошибка при конвертации датасета '{dataset_key}': {e}", file=sys.stderr)
        raise

    print(f"wrote {os.path.join(args.outdir, 'feature.txt')}")
    print(f"wrote {os.path.join(args.outdir, 'net.txt')}")
    print(f"wrote {os.path.join(args.outdir, 'label.txt')}")
    print("wrote train.txt, dev.txt, test.txt")

    split_kind = "официальный (PyG)" if stats["used_official_split"] else "сгенерированный (НЕ официальный)"
    print()
    print("=" * 60)
    print(f"Сводка: {dataset_key}")
    print(f"  узлов:             {stats['n_nodes']}")
    print(f"  фичей:             {stats['n_feats']}")
    print(f"  классов:           {stats['n_classes']}")
    print(f"  рёбер (net.txt):   {stats['n_edges']}")
    print(f"  train / dev / test: {stats['n_train']} / {stats['n_dev']} / {stats['n_test']}")
    print(f"  тип сплита:        {split_kind}")
    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
