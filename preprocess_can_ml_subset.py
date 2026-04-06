"""
Preprocessing for CAN-ML raw CSV files -> sequence tensors for multi-class classification
PT VERSION:
- Create ALL windows first, then split train/val/test
- Stratified split by window labels so all classes are preserved much better
- Optional quick subset while preserving class ratios
- Optional undersampling of normal class in training set
- Save outputs as .pt for PyTorch

Assumptions
-----------
- Each CSV file corresponds to one attack type.
- Column `attack` is binary:
    0 = normal
    1 = attack of that file's type
- Final labels:
    class 0 = normal
    class i = attack type of file i (1..N)

Expected CSV columns
--------------------
timestamp, arbitration_id, data_field, attack
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


@dataclass
class PreprocessConfig:
    input_dir: str
    output_dir: str

    window_size: int = 64
    stride: int = 4

    train_ratio: float = 0.70
    val_ratio: float = 0.10
    test_ratio: float = 0.20

    freq_history: int = 128
    min_frames_per_file: int = 1000
    dtype: str = "float32"
    label_rule: str = "any_attack"

    # Quick experiment controls
    subset_fraction: float = 1.0
    max_train_windows: Optional[int] = None
    max_val_windows: Optional[int] = None
    max_test_windows: Optional[int] = None
    random_seed: int = 42

    # Optional: reduce dominance of normal class in training set only
    undersample_normal_train: bool = False
    normal_to_attack_ratio: float = 3.0


def safe_hex_to_bytes(hex_str: str, max_len: int = 8) -> Tuple[np.ndarray, int]:
    if pd.isna(hex_str):
        return np.zeros(max_len, dtype=np.float32), 0

    s = str(hex_str).strip().lower().replace(" ", "")
    if s.startswith("0x"):
        s = s[2:]
    if len(s) % 2 == 1:
        s = "0" + s
    s = s[: max_len * 2]

    vals = []
    for i in range(0, len(s), 2):
        try:
            vals.append(int(s[i:i+2], 16))
        except ValueError:
            vals.append(0)

    dlc = len(vals)
    if dlc < max_len:
        vals.extend([0] * (max_len - dlc))

    return np.array(vals, dtype=np.float32), dlc


def add_basic_features(df: pd.DataFrame, freq_history: int) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    df["arbitration_id"] = pd.to_numeric(df["arbitration_id"], errors="coerce").fillna(0).astype(np.int64)
    df["attack"] = pd.to_numeric(df["attack"], errors="coerce").fillna(0).astype(np.int64)

    payload_arr = np.zeros((len(df), 8), dtype=np.float32)
    dlc_arr = np.zeros(len(df), dtype=np.int64)

    for i, s in enumerate(df["data_field"].values):
        payload, dlc = safe_hex_to_bytes(s, max_len=8)
        payload_arr[i] = payload
        dlc_arr[i] = dlc

    for j in range(8):
        df[f"payload_{j}"] = payload_arr[:, j]
    df["dlc"] = dlc_arr

    ts = pd.to_numeric(df["timestamp"], errors="coerce").ffill().bfill().values.astype(np.float64)
    delta_t = np.zeros(len(df), dtype=np.float64)
    if len(df) > 1:
        delta_t[1:] = np.maximum(ts[1:] - ts[:-1], 0.0)
    df["delta_t"] = delta_t
    df["log_delta_t"] = np.log1p(delta_t)

    ids = df["arbitration_id"].values
    freq_recent = np.zeros(len(df), dtype=np.float32)
    counts: Dict[int, int] = {}
    queue: List[int] = []

    for i, can_id in enumerate(ids):
        can_id = int(can_id)
        queue.append(can_id)
        counts[can_id] = counts.get(can_id, 0) + 1

        if len(queue) > freq_history:
            old = queue.pop(0)
            counts[old] -= 1
            if counts[old] == 0:
                del counts[old]

        freq_recent[i] = counts.get(can_id, 0)

    df["id_freq_recent"] = freq_recent
    return df


def build_scalers_from_frames(df_all: pd.DataFrame) -> Dict[str, MinMaxScaler]:
    scalers = {
        "arbitration_id": MinMaxScaler(),
        "dlc": MinMaxScaler(),
        "log_delta_t": MinMaxScaler(),
        "id_freq_recent": MinMaxScaler(),
    }
    scalers["arbitration_id"].fit(df_all[["arbitration_id"]].values.astype(np.float32))
    scalers["dlc"].fit(df_all[["dlc"]].values.astype(np.float32))
    scalers["log_delta_t"].fit(df_all[["log_delta_t"]].values.astype(np.float32))
    scalers["id_freq_recent"].fit(df_all[["id_freq_recent"]].values.astype(np.float32))
    return scalers


def transform_frames(df: pd.DataFrame, scalers: Dict[str, MinMaxScaler]) -> np.ndarray:
    id_norm = scalers["arbitration_id"].transform(df[["arbitration_id"]].values.astype(np.float32))
    dlc_norm = scalers["dlc"].transform(df[["dlc"]].values.astype(np.float32))
    payload = df[[f"payload_{j}" for j in range(8)]].values.astype(np.float32) / 255.0
    log_dt_norm = scalers["log_delta_t"].transform(df[["log_delta_t"]].values.astype(np.float32))
    freq_norm = scalers["id_freq_recent"].transform(df[["id_freq_recent"]].values.astype(np.float32))
    return np.concatenate([id_norm, dlc_norm, payload, log_dt_norm, freq_norm], axis=1).astype(np.float32)


def make_windows(
    features: np.ndarray,
    attack_binary: np.ndarray,
    class_id: int,
    window_size: int,
    stride: int,
    label_rule: str = "any_attack",
) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    n = len(features)
    if n < window_size:
        return np.empty((0, window_size, features.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int64)

    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        xw = features[start:end]
        aw = attack_binary[start:end]

        if label_rule == "any_attack":
            y = class_id if np.any(aw == 1) else 0
        elif label_rule == "last_frame":
            y = class_id if aw[-1] == 1 else 0
        else:
            raise ValueError(f"Unsupported label_rule: {label_rule}")

        xs.append(xw)
        ys.append(y)

    return np.stack(xs, axis=0).astype(np.float32), np.asarray(ys, dtype=np.int64)


def stratified_subsample(
    X: np.ndarray,
    y: np.ndarray,
    fraction: float,
    max_samples: Optional[int],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(X) == 0:
        return X, y
    if not (0 < fraction <= 1.0):
        raise ValueError("subset_fraction must be in (0,1].")

    n_total = len(y)
    target_n = int(round(n_total * fraction))
    if max_samples is not None:
        target_n = min(target_n, max_samples)
    target_n = max(1, target_n)

    if target_n >= n_total:
        return X, y

    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    raw_targets = counts / counts.sum() * target_n
    alloc = np.floor(raw_targets).astype(int)

    if target_n >= len(classes):
        alloc = np.maximum(alloc, 1)

    while alloc.sum() > target_n:
        idx = np.argmax(alloc)
        if alloc[idx] > 1:
            alloc[idx] -= 1
        else:
            break

    remainder = raw_targets - np.floor(raw_targets)
    while alloc.sum() < target_n:
        idx = int(np.argmax(remainder))
        if alloc[idx] < counts[idx]:
            alloc[idx] += 1
        remainder[idx] = -1.0

    selected = []
    for cls, need in zip(classes, alloc):
        cls_idx = np.where(y == cls)[0]
        need = min(int(need), len(cls_idx))
        if need > 0:
            selected.append(rng.choice(cls_idx, size=need, replace=False))

    selected = np.concatenate(selected)
    rng.shuffle(selected)
    return X[selected], y[selected]


def undersample_normal(
    X: np.ndarray,
    y: np.ndarray,
    normal_to_attack_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(X) == 0:
        return X, y

    idx_normal = np.where(y == 0)[0]
    idx_attack = np.where(y != 0)[0]

    if len(idx_normal) == 0 or len(idx_attack) == 0:
        return X, y

    max_normal = int(round(normal_to_attack_ratio * len(idx_attack)))
    if len(idx_normal) <= max_normal:
        return X, y

    rng = np.random.default_rng(seed)
    keep_normal = rng.choice(idx_normal, size=max_normal, replace=False)
    keep_idx = np.concatenate([keep_normal, idx_attack])
    rng.shuffle(keep_idx)
    return X[keep_idx], y[keep_idx]


def label_distribution(y: np.ndarray) -> Dict[int, int]:
    if len(y) == 0:
        return {}
    vals, cnts = np.unique(y, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, cnts)}


def save_metadata(out_dir: Path, cfg: PreprocessConfig, class_map: Dict[str, int], feature_dim: int, stats: Dict):
    meta = {
        "config": cfg.__dict__,
        "class_map": class_map,
        "feature_description": [
            "arbitration_id_norm", "dlc_norm",
            "payload_0_norm255", "payload_1_norm255", "payload_2_norm255", "payload_3_norm255",
            "payload_4_norm255", "payload_5_norm255", "payload_6_norm255", "payload_7_norm255",
            "log_delta_t_norm", "id_freq_recent_norm",
        ],
        "feature_dim": feature_dim,
        "stats": stats,
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def save_split_pt(path: Path, X: np.ndarray, y: np.ndarray):
    obj = {
        "X": torch.from_numpy(X).float(),
        "y": torch.from_numpy(y).long(),
    }
    torch.save(obj, path)


def preprocess_can_multiclass(cfg: PreprocessConfig):
    input_dir = Path(cfg.input_dir)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.glob("*.csv"))
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    print(f"Found {len(csv_files)} CSV files.")
    class_map = {csv_path.name: i + 1 for i, csv_path in enumerate(csv_files)}

    per_file_frames = []
    for csv_path in csv_files:
        class_id = class_map[csv_path.name]
        df = pd.read_csv(csv_path)
        required_cols = {"timestamp", "arbitration_id", "data_field", "attack"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path.name} missing required columns: {missing}")

        df = add_basic_features(df, freq_history=cfg.freq_history)
        if len(df) < cfg.min_frames_per_file:
            raise ValueError(f"{csv_path.name} has only {len(df)} frames, smaller than min_frames_per_file={cfg.min_frames_per_file}")

        per_file_frames.append({
            "file_name": csv_path.name,
            "class_id": class_id,
            "df": df,
        })

    all_frames = pd.concat([item["df"] for item in per_file_frames], axis=0, ignore_index=True)
    scalers = build_scalers_from_frames(all_frames)

    X_all_list, y_all_list = [], []
    per_file_stats = []

    for item in per_file_frames:
        df = item["df"]
        class_id = item["class_id"]
        file_name = item["file_name"]

        feats = transform_frames(df, scalers)
        attack_bin = df["attack"].values.astype(np.int64)

        X_file, y_file = make_windows(
            features=feats,
            attack_binary=attack_bin,
            class_id=class_id,
            window_size=cfg.window_size,
            stride=cfg.stride,
            label_rule=cfg.label_rule,
        )

        X_all_list.append(X_file)
        y_all_list.append(y_file)
        per_file_stats.append({
            "file_name": file_name,
            "class_id": class_id,
            "frames": int(len(df)),
            "windows": int(len(X_file)),
            "label_dist": label_distribution(y_file),
        })

    X_all = np.concatenate(X_all_list, axis=0).astype(cfg.dtype)
    y_all = np.concatenate(y_all_list, axis=0).astype(np.int64)

    print(f"Total windows before split: {X_all.shape}")

    temp_ratio = cfg.val_ratio + cfg.test_ratio
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all, y_all,
        test_size=temp_ratio,
        random_state=cfg.random_seed,
        stratify=y_all,
        shuffle=True,
    )

    test_ratio_in_temp = cfg.test_ratio / (cfg.val_ratio + cfg.test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=test_ratio_in_temp,
        random_state=cfg.random_seed + 1,
        stratify=y_temp,
        shuffle=True,
    )

    X_train, y_train = stratified_subsample(X_train, y_train, cfg.subset_fraction, cfg.max_train_windows, cfg.random_seed)
    X_val, y_val = stratified_subsample(X_val, y_val, cfg.subset_fraction, cfg.max_val_windows, cfg.random_seed + 1)
    X_test, y_test = stratified_subsample(X_test, y_test, cfg.subset_fraction, cfg.max_test_windows, cfg.random_seed + 2)

    if cfg.undersample_normal_train:
        X_train, y_train = undersample_normal(
            X_train, y_train,
            normal_to_attack_ratio=cfg.normal_to_attack_ratio,
            seed=cfg.random_seed + 3,
        )

    rng = np.random.default_rng(cfg.random_seed)
    perm = rng.permutation(len(X_train))
    X_train = X_train[perm]
    y_train = y_train[perm]

    save_split_pt(out_dir / "train.pt", X_train, y_train)
    save_split_pt(out_dir / "val.pt", X_val, y_val)
    save_split_pt(out_dir / "test.pt", X_test, y_test)

    stats = {
        "num_classes_total": len(csv_files) + 1,
        "subset_fraction": cfg.subset_fraction,
        "max_train_windows": cfg.max_train_windows,
        "max_val_windows": cfg.max_val_windows,
        "max_test_windows": cfg.max_test_windows,
        "undersample_normal_train": cfg.undersample_normal_train,
        "normal_to_attack_ratio": cfg.normal_to_attack_ratio,
        "per_file": per_file_stats,
        "train_label_dist": label_distribution(y_train),
        "val_label_dist": label_distribution(y_val),
        "test_label_dist": label_distribution(y_test),
    }

    save_metadata(out_dir, cfg, class_map, X_train.shape[-1], stats)

    print("\nSaved:")
    print(f"  {out_dir / 'train.pt'}  X={tuple(X_train.shape)}, y={tuple(y_train.shape)}")
    print(f"  {out_dir / 'val.pt'}    X={tuple(X_val.shape)}, y={tuple(y_val.shape)}")
    print(f"  {out_dir / 'test.pt'}   X={tuple(X_test.shape)}, y={tuple(y_test.shape)}")
    print(f"  {out_dir / 'metadata.json'}")

    def show_dist(name, y):
        vals, cnts = np.unique(y, return_counts=True)
        total = len(y)
        print(f"{name} label distribution:")
        for v, c in zip(vals, cnts):
            print(f"  class {int(v):2d}: {int(c):6d}  ({100.0 * c / total:6.2f}%)")

    print()
    show_dist("Train", y_train)
    show_dist("Val", y_val)
    show_dist("Test", y_test)


if __name__ == "__main__":
    cfg = PreprocessConfig(
        input_dir="./data/2017-subaru-forester/merged",
        output_dir="./data/2017-subaru-forester/preprocessed",
        window_size=64,
        stride=32,
        train_ratio=0.70,
        val_ratio=0.10,
        test_ratio=0.20,
        freq_history=128,
        min_frames_per_file=1000,
        label_rule="any_attack",
        subset_fraction=1.0,
        max_train_windows=None,
        max_val_windows=None,
        max_test_windows=None,
        random_seed=42,
        undersample_normal_train=True,
        normal_to_attack_ratio=3.0,
    )
    preprocess_can_multiclass(cfg)
