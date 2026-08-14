import os
import numpy as np
import random

dir = r"D:\PythonProject\CGNN_Redux\dataset\pubmed"

os.makedirs(dir, exist_ok=True)

node_file = os.path.join(dir, "Pubmed-Diabetes.NODE.paper.tab")
cites_file = os.path.join(dir, "Pubmed-Diabetes.DIRECTED.cites.tab")

# ---------- 1. Чтение узлов ----------
def parse_node_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    data_lines = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split('\t')
        if len(parts) < 3:
            parts = line.split()
        if len(parts) >= 3:
            data_lines.append(parts)
    if not data_lines:
        raise ValueError("No data lines in node file.")
    max_cols = max(len(row) for row in data_lines)
    print(f"Max columns in node file: {max_cols}")
    if max_cols == 3 and any(':' in row[2] for row in data_lines):
        print("Sparse format detected (word:value pairs).")
        ids, labels_raw, vocab = [], [], {}
        features_sparse = []
        for row in data_lines:
            node_id, label, feat_str = row[0], row[1], row[2]
            pairs = feat_str.split()
            feat_dict = {}
            for pair in pairs:
                if ':' not in pair:
                    continue
                word, val = pair.split(':')
                if word not in vocab:
                    vocab[word] = len(vocab)
                feat_dict[vocab[word]] = float(val)
            ids.append(node_id)
            labels_raw.append(label)
            features_sparse.append(feat_dict)
        num_features = len(vocab)
        num_nodes = len(ids)
        feat_matrix = np.zeros((num_nodes, num_features), dtype=float)
        for i, fdict in enumerate(features_sparse):
            for fid, val in fdict.items():
                feat_matrix[i, fid] = val
        return ids, labels_raw, feat_matrix
    else:
        print("Dense format detected (each column is a feature).")
        ids = [row[0] for row in data_lines]
        labels_raw = [row[1] for row in data_lines]
        feat_matrix = []
        for row in data_lines:
            feat_row = []
            for val in row[2:]:
                try:
                    feat_row.append(float(val))
                except ValueError:
                    feat_row.append(0.0)
            while len(feat_row) < max_cols - 2:
                feat_row.append(0.0)
            feat_matrix.append(feat_row)
        feat_matrix = np.array(feat_matrix, dtype=float)
        return ids, labels_raw, feat_matrix

ids, labels_raw, feat_matrix = parse_node_file(node_file)
unique_labels = sorted(set(labels_raw))
label_to_int = {lbl: i for i, lbl in enumerate(unique_labels)}
labels = np.array([label_to_int[lbl] for lbl in labels_raw], dtype=int)
num_nodes = len(ids)
node_to_idx = {nid: i for i, nid in enumerate(ids)}
num_features = feat_matrix.shape[1]
print(f"Nodes: {num_nodes}, Features: {num_features}, Classes: {len(unique_labels)}")

# Сохраняем feature.txt и label.txt
with open(os.path.join(dir, 'feature.txt'), 'w') as f:
    for i in range(num_nodes):
        row = feat_matrix[i]
        non_zero = np.nonzero(row)[0]
        feat_pairs = ' '.join([f"{j}:{row[j]}" for j in non_zero if row[j] != 0.0])
        f.write(f"{i}\t{feat_pairs}\n")

with open(os.path.join(dir, 'label.txt'), 'w') as f:
    for i, lbl in enumerate(labels):
        f.write(f"{i}\t{lbl}\n")

# ---------- 2. Чтение рёбер (правильный парсинг) ----------
edges = []
with open(cites_file, 'r') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('DIRECTED') or line.startswith('NO_FEATURES'):
            continue
        # Пример строки: "33824	paper:19127292	|	paper:17363749"
        parts = line.split('\t')
        if len(parts) >= 4:
            u_str = parts[1].replace('paper:', '')
            v_str = parts[3].replace('paper:', '')
            if u_str in node_to_idx and v_str in node_to_idx:
                edges.append((node_to_idx[u_str], node_to_idx[v_str]))
        elif len(parts) == 3 and '|' in parts[1]:
            # Возможен вариант без табуляции между | и идентификаторами
            token = parts[1].split('|')
            if len(token) == 2:
                u_str = token[0].strip().replace('paper:', '')
                v_str = token[1].strip().replace('paper:', '')
                if u_str in node_to_idx and v_str in node_to_idx:
                    edges.append((node_to_idx[u_str], node_to_idx[v_str]))

print(f"Total edges found: {len(edges)}")
if len(edges) > 0:
    print(f"Sample edges (first 5): {edges[:5]}")
else:
    print("WARNING: No edges found. net.txt will be empty.")

# Сохраняем net.txt
with open(os.path.join(dir, 'net.txt'), 'w') as f:
    for u, v in edges:
        f.write(f"{u}\t{v}\n")

# ---------- 3. Разбиение ----------
random.seed(42)
num_classes = len(unique_labels)
idx_by_label = {lbl: [] for lbl in range(num_classes)}
for i, lbl in enumerate(labels):
    idx_by_label[lbl].append(i)

train_idx = []
for lbl in range(num_classes):
    num_train = min(20, len(idx_by_label[lbl]))
    selected = random.sample(idx_by_label[lbl], num_train)
    train_idx.extend(selected)

remaining = list(set(range(num_nodes)) - set(train_idx))
random.shuffle(remaining)
dev_idx = remaining[:500]
test_idx = remaining[500:]

with open(os.path.join(dir, 'train.txt'), 'w') as f:
    for idx in train_idx:
        f.write(str(idx) + '\n')
with open(os.path.join(dir, 'dev.txt'), 'w') as f:
    for idx in dev_idx:
        f.write(str(idx) + '\n')
with open(os.path.join(dir, 'test.txt'), 'w') as f:
    for idx in test_idx:
        f.write(str(idx) + '\n')

print("Датасет готов.")