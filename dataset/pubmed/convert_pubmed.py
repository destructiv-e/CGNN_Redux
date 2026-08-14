import os
import random
import logging
from typing import List, Tuple, Dict

import numpy as np

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_node_file(filepath: str) -> Tuple[List[str], List[str], np.ndarray]:
    """
    Парсит файл узлов PubMed-Diabetes.
    Поддерживает как разреженный (word:value), так и плотный форматы.
    В разреженном формате ключом считается всё, что до последнего двоеточия,
    а значением — после последнего двоеточия (например, "numeric:w-common:0.0").
    """
    logger.info(f"Начинаем парсинг файла узлов: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    if not lines:
        raise ValueError("Файл узлов пуст или не содержит данных.")

    data_rows = []
    for line in lines:
        parts = line.split('\t')
        if len(parts) < 3:
            parts = line.split()
        if len(parts) >= 3:
            data_rows.append(parts)

    if not data_rows:
        raise ValueError("Не удалось разобрать данные в файле узлов.")

    max_cols = max(len(row) for row in data_rows)
    logger.info(f"Максимальное число колонок в файле узлов: {max_cols}")

    is_sparse = any(':' in row[2] for row in data_rows)

    if is_sparse:
        logger.info("Обнаружен разреженный формат (word:value).")
        ids = []
        labels_raw = []
        vocab: Dict[str, int] = {}
        features_sparse: List[Dict[int, float]] = []

        for row in data_rows:
            node_id, label, feat_str = row[0], row[1], row[2]
            ids.append(node_id)
            labels_raw.append(label)

            feat_dict = {}
            for pair in feat_str.split():
                if ':' not in pair:
                    continue
                # Ищем последнее двоеточие, чтобы отделить ключ (может содержать ':') от значения
                last_colon = pair.rfind(':')
                if last_colon == -1:
                    continue
                word = pair[:last_colon]
                val_str = pair[last_colon+1:]
                try:
                    val = float(val_str)
                except ValueError:
                    logger.warning(f"Некорректное значение в паре: {pair}, пропускаем.")
                    continue
                if word not in vocab:
                    vocab[word] = len(vocab)
                feat_dict[vocab[word]] = val
            features_sparse.append(feat_dict)

        num_features = len(vocab)
        num_nodes = len(ids)
        feat_matrix = np.zeros((num_nodes, num_features), dtype=float)
        for i, fdict in enumerate(features_sparse):
            for fid, val in fdict.items():
                feat_matrix[i, fid] = val

        logger.info(f"Создана матрица признаков размером {num_nodes}x{num_features} (разреженная).")
        return ids, labels_raw, feat_matrix

    else:
        logger.info("Обнаружен плотный формат (каждая колонка — признак).")
        ids = [row[0] for row in data_rows]
        labels_raw = [row[1] for row in data_rows]

        feat_matrix = []
        for row in data_rows:
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
        logger.info(f"Создана матрица признаков размером {feat_matrix.shape} (плотная).")
        return ids, labels_raw, feat_matrix


def parse_edges_file(filepath: str, node_to_idx: Dict[str, int]) -> List[Tuple[int, int]]:
    logger.info(f"Начинаем парсинг файла рёбер: {filepath}")
    edges = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('DIRECTED') or line.startswith('NO_FEATURES'):
                continue
            parts = line.split('\t')
            if len(parts) >= 4:
                u_str = parts[1].replace('paper:', '').strip()
                v_str = parts[3].replace('paper:', '').strip()
            elif len(parts) == 3 and '|' in parts[1]:
                token = parts[1].split('|')
                if len(token) == 2:
                    u_str = token[0].replace('paper:', '').strip()
                    v_str = token[1].replace('paper:', '').strip()
                else:
                    continue
            else:
                continue
            if u_str in node_to_idx and v_str in node_to_idx:
                edges.append((node_to_idx[u_str], node_to_idx[v_str]))
            else:
                logger.warning(f"Пропущено ребро с неизвестными узлами: {u_str} -> {v_str}")
    logger.info(f"Найдено рёбер: {len(edges)}")
    if edges:
        logger.info(f"Пример первых 5 рёбер: {edges[:5]}")
    else:
        logger.warning("Рёбра не найдены. Файл net.txt будет пустым.")
    return edges


def split_data(labels: np.ndarray, num_nodes: int, train_per_class: int = 20,
               dev_size: int = 500, random_seed: int = 42) -> Tuple[List[int], List[int], List[int]]:
    logger.info("Начинаем разбиение данных.")
    random.seed(random_seed)
    num_classes = int(np.max(labels) + 1)
    idx_by_label = {cls: [] for cls in range(num_classes)}
    for i, lbl in enumerate(labels):
        idx_by_label[lbl].append(i)

    train_idx = []
    for cls in range(num_classes):
        available = idx_by_label[cls]
        n_train = min(train_per_class, len(available))
        selected = random.sample(available, n_train)
        train_idx.extend(selected)

    remaining = list(set(range(num_nodes)) - set(train_idx))
    random.shuffle(remaining)
    dev_idx = remaining[:dev_size]
    test_idx = remaining[dev_size:]
    logger.info(f"Разбиение: train={len(train_idx)}, dev={len(dev_idx)}, test={len(test_idx)}")
    return train_idx, dev_idx, test_idx


def save_data(data_dir: str, feat_matrix: np.ndarray, labels: np.ndarray,
              edges: List[Tuple[int, int]], train_idx: List[int],
              dev_idx: List[int], test_idx: List[int]) -> None:
    os.makedirs(data_dir, exist_ok=True)
    logger.info(f"Сохраняем данные в директорию: {data_dir}")

    feature_path = os.path.join(data_dir, 'feature.txt')
    with open(feature_path, 'w', encoding='utf-8') as f:
        for i in range(feat_matrix.shape[0]):
            row = feat_matrix[i]
            non_zero = np.nonzero(row)[0]
            feat_pairs = ' '.join(f"{j}:{row[j]}" for j in non_zero if row[j] != 0.0)
            f.write(f"{i}\t{feat_pairs}\n")
    logger.info(f"Сохранён {feature_path}")

    label_path = os.path.join(data_dir, 'label.txt')
    with open(label_path, 'w', encoding='utf-8') as f:
        for i, lbl in enumerate(labels):
            f.write(f"{i}\t{lbl}\n")
    logger.info(f"Сохранён {label_path}")

    net_path = os.path.join(data_dir, 'net.txt')
    with open(net_path, 'w', encoding='utf-8') as f:
        for u, v in edges:
            f.write(f"{u}\t{v}\n")
    logger.info(f"Сохранён {net_path} (рёбер: {len(edges)})")

    for name, idx_list in [('train', train_idx), ('dev', dev_idx), ('test', test_idx)]:
        path = os.path.join(data_dir, f'{name}.txt')
        with open(path, 'w', encoding='utf-8') as f:
            for idx in idx_list:
                f.write(f"{idx}\n")
        logger.info(f"Сохранён {path} (записей: {len(idx_list)})")


def main():
    data_dir = r"D:\PythonProject\CGNN_Redux\dataset\pubmed"
    node_file = os.path.join(data_dir, "Pubmed-Diabetes.NODE.paper.tab")
    cites_file = os.path.join(data_dir, "Pubmed-Diabetes.DIRECTED.cites.tab")

    if not os.path.exists(node_file):
        raise FileNotFoundError(f"Файл узлов не найден: {node_file}")
    if not os.path.exists(cites_file):
        raise FileNotFoundError(f"Файл рёбер не найден: {cites_file}")

    ids, labels_raw, feat_matrix = parse_node_file(node_file)

    unique_labels = sorted(set(labels_raw))
    label_to_int = {lbl: i for i, lbl in enumerate(unique_labels)}
    labels = np.array([label_to_int[lbl] for lbl in labels_raw], dtype=int)
    num_nodes = len(ids)
    num_features = feat_matrix.shape[1]
    num_classes = len(unique_labels)

    logger.info(f"Итоги по узлам: узлов={num_nodes}, признаков={num_features}, классов={num_classes}")

    node_to_idx = {nid: i for i, nid in enumerate(ids)}
    edges = parse_edges_file(cites_file, node_to_idx)

    train_idx, dev_idx, test_idx = split_data(labels, num_nodes)
    save_data(data_dir, feat_matrix, labels, edges, train_idx, dev_idx, test_idx)

    logger.info("Датасет успешно подготовлен.")


if __name__ == "__main__":
    main()