#!/usr/bin/env python3
"""
convert_pubmed_raw.py

Превращает "сырые" файлы датасета Pubmed-Diabetes (формат LINQS / из архива
Pubmed-Diabetes.tar.gz) в готовый набор файлов того же формата, что
используется вашей моделью:

    feature.txt   id \t idx:val idx:val ...
    net.txt       src \t dst \t 1        (рёбра продублированы в обе стороны)
    label.txt     id \t label            (только узлы train+dev+test)
    train.txt     id                     (по одному узлу на строку)
    dev.txt       id
    test.txt      id

Входные файлы (стандартные имена из архива LINQS):
    Pubmed-Diabetes_NODE_paper.tab
    Pubmed-Diabetes_DIRECTED_cites.tab
    Pubmed-Diabetes_GRAPH_pubmed.tab   (не используется, но проверяется на наличие)

Использование:
    python convert_pubmed_raw.py \
        --node Pubmed-Diabetes_NODE_paper.tab \
        --cites Pubmed-Diabetes_DIRECTED_cites.tab \
        --outdir ./out \
        --train-per-class 20 --dev-size 500 --test-size 1000 --seed 42

Важно про сплит (train/dev/test):
    В официальном датасете (Planetoid / Kipf GCN) разбиение на train/dev/test
    было зафиксировано авторами и "зашито" в бинарных .pkl-файлах — из "сырых"
    .tab-файлов эту информацию восстановить нельзя. Поэтому здесь сплит
    строится по тому же общепринятому рецепту (20 узлов на класс в train,
    затем 500 в dev, затем 1000 в test), но со своим фиксированным seed —
    то есть состав train/dev/test будет отличаться от классического
    Planetoid-сплита (хотя размеры совпадают: 60/500/1000, 3 класса).
    Если вам нужен побитово тот же сплит, что и в официальном Planetoid-
    датасете, используйте не эти сырые файлы, а ind.pubmed.* (как в прошлый
    раз).
"""

import argparse
import random
from collections import defaultdict


def parse_node_file(path):
    """Parse Pubmed-Diabetes_NODE_paper.tab.

    Returns:
        paper_ids: list of paper ids in file order (defines node index 0..N-1)
        labels: dict paper_id -> int label (as given in file, e.g. 1/2/3)
        feat_name_to_idx: dict feature name ("w-xxx") -> global feature index
        features: dict paper_id -> list of (feat_idx, value)
    """
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    # line 0: "NODE\tpaper"
    # line 1: header with feature names
    header = lines[1].rstrip("\n").split("\t")

    feat_name_to_idx = {}
    for field in header:
        # fields look like "numeric:w-rat:0.0", "cat=1,2,3:label", "string:summary"
        if field.startswith("numeric:"):
            name = field.split(":")[1]
            feat_name_to_idx[name] = len(feat_name_to_idx)

    paper_ids = []
    labels = {}
    features = {}

    for line in lines[2:]:
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split("\t")
        pid = parts[0]
        row_label = None
        row_feats = []
        for field in parts[1:]:
            if field.startswith("summary="):
                continue
            if "=" not in field:
                continue
            key, val = field.split("=", 1)
            if key == "label":
                row_label = int(val)
            elif key in feat_name_to_idx:
                row_feats.append((feat_name_to_idx[key], float(val)))
            # unknown keys (shouldn't happen) are ignored

        paper_ids.append(pid)
        labels[pid] = row_label
        features[pid] = row_feats

    return paper_ids, labels, feat_name_to_idx, features


def parse_cites_file(path):
    """Parse Pubmed-Diabetes_DIRECTED_cites.tab.

    Returns list of (src_paper_id, dst_paper_id) directed citation edges.
    """
    edges = []
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    # line 0: "DIRECTED\tcites"
    # line 1: "NO_FEATURES"
    for line in lines[2:]:
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split("\t")
        # parts: [edge_id, "paper:SRC", "|", "paper:DST"]
        src = parts[1].split(":", 1)[1]
        dst = parts[3].split(":", 1)[1]
        edges.append((src, dst))
    return edges


def build_split(labels_by_node, train_per_class, dev_size, test_size, seed):
    """Stratified train split (N per class) + random dev/test from the rest.

    labels_by_node: dict node_idx -> label
    Returns (train_ids, dev_ids, test_ids) each a sorted-by-selection list of
    node indices (train_ids ordered by class then shuffle order, to mirror
    the classic Planetoid layout).
    """
    rng = random.Random(seed)

    by_class = defaultdict(list)
    for node_idx, lbl in labels_by_node.items():
        by_class[lbl].append(node_idx)
    for lbl in by_class:
        rng.shuffle(by_class[lbl])

    train_ids = []
    used = set()
    for lbl in sorted(by_class):
        chosen = by_class[lbl][:train_per_class]
        train_ids.extend(chosen)
        used.update(chosen)

    remaining = [n for n in labels_by_node if n not in used]
    rng.shuffle(remaining)

    dev_ids = remaining[:dev_size]
    test_ids = remaining[dev_size:dev_size + test_size]

    return train_ids, dev_ids, test_ids


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--node", required=True, help="path to Pubmed-Diabetes_NODE.paper.tab")
    ap.add_argument("--cites", required=True, help="path to Pubmed-Diabetes_DIRECTED.cites.tab")
    ap.add_argument("--outdir", default=".", help="output directory")
    ap.add_argument("--train-per-class", type=int, default=20)
    ap.add_argument("--dev-size", type=int, default=500)
    ap.add_argument("--test-size", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--float-fmt", default="{:.6g}", help="format string for feature values")
    args = ap.parse_args()

    import os
    os.makedirs(args.outdir, exist_ok=True)

    print("Reading node/feature/label file...")
    paper_ids, raw_labels, feat_name_to_idx, raw_features = parse_node_file(args.node)
    n_nodes = len(paper_ids)
    n_feats = len(feat_name_to_idx)
    print(f"  nodes: {n_nodes}, feature dim: {n_feats}")

    # paper_id -> node index (0..N-1), in file order
    pid_to_idx = {pid: i for i, pid in enumerate(paper_ids)}

    # remap raw labels (1,2,3,...) -> 0-based contiguous ints, sorted
    unique_labels = sorted(set(raw_labels.values()))
    label_remap = {old: new for new, old in enumerate(unique_labels)}
    print(f"  classes found: {unique_labels} -> remapped to {list(label_remap.values())}")

    labels_by_node = {pid_to_idx[pid]: label_remap[raw_labels[pid]] for pid in paper_ids}

    print("Reading citation edges...")
    cite_edges = parse_cites_file(args.cites)
    print(f"  directed citation records: {len(cite_edges)}")

    edges = set()
    skipped = 0
    for src, dst in cite_edges:
        if src not in pid_to_idx or dst not in pid_to_idx:
            skipped += 1
            continue
        u, v = pid_to_idx[src], pid_to_idx[dst]
        if u == v:
            continue
        edges.add((u, v))
        edges.add((v, u))
    if skipped:
        print(f"  warning: skipped {skipped} edges referencing unknown paper ids")
    print(f"  undirected edges written (both directions): {len(edges)}")

    print("Building split (train/dev/test)...")
    train_ids, dev_ids, test_ids = build_split(
        labels_by_node, args.train_per_class, args.dev_size, args.test_size, args.seed
    )
    print(f"  train: {len(train_ids)}  dev: {len(dev_ids)}  test: {len(test_ids)}")

    # ---- write feature.txt ----
    feat_path = os.path.join(args.outdir, "feature.txt")
    with open(feat_path, "w", encoding="utf-8") as f:
        for pid in paper_ids:
            idx = pid_to_idx[pid]
            feats = sorted(raw_features[pid], key=lambda t: t[0])
            parts = [f"{fi}:{args.float_fmt.format(val)}" for fi, val in feats]
            f.write(f"{idx}\t" + " ".join(parts) + "\n")
    print(f"wrote {feat_path}")

    # ---- write net.txt ----
    net_path = os.path.join(args.outdir, "net.txt")
    with open(net_path, "w", encoding="utf-8") as f:
        for u, v in sorted(edges):
            f.write(f"{u}\t{v}\t1\n")
    print(f"wrote {net_path}")

    # ---- write label.txt (train + dev + test ids, in that order) ----
    label_path = os.path.join(args.outdir, "label.txt")
    all_split_ids = train_ids + dev_ids + test_ids
    with open(label_path, "w", encoding="utf-8") as f:
        for idx in all_split_ids:
            f.write(f"{idx}\t{labels_by_node[idx]}\n")
    print(f"wrote {label_path}")

    # ---- write train.txt / dev.txt / test.txt ----
    def write_ids(path, ids):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(str(i) for i in ids) + "\n")

    write_ids(os.path.join(args.outdir, "train.txt"), train_ids)
    write_ids(os.path.join(args.outdir, "dev.txt"), dev_ids)
    write_ids(os.path.join(args.outdir, "test.txt"), test_ids)
    print("wrote train.txt, dev.txt, test.txt")

    print("Done.")


if __name__ == "__main__":
    main()