"""
Preprocessing for CAN-ML raw CSV files -> sequence tensors for multi-class classification

Assumptions
-----------
- The input folder contains 9 CSV files.
- Each CSV file contains BOTH normal frames and attack frames.
- The `attack` column is binary:
      0 = normal
      1 = attack
- Each CSV corresponds to ONE attack type.
  Therefore:
      normal frames -> class 0
      attack frames in file i -> class i (1..9)

Expected CSV columns
--------------------
timestamp, arbitration_id, data_field, attack

Example
-------
timestamp,arbitration_id,data_field,attack
1672531200.0,164,0200000000000002,0
1672531200.0,149,160c3f1fa9150000,0
...

What this script does
---------------------
1. Reads all CSVs in a folder
2. Sorts each file by timestamp
3. Builds per-frame features:
   - normalized arbitration_id
   - dlc
   - payload bytes (8 bytes)
   - log1p(delta_t)
   - local ID frequency in recent history
4. Splits each file by time/order into train/val/test with guard bands
5. Builds sliding windows
6. Assigns a multi-class label to each window:
   - if no attack in the window -> class 0
   - otherwise -> class ID of that file
7. Fits scalers ONLY on training frames
8. Saves:
   - X_train.npy, y_train.npy
   - X_val.npy,   y_val.npy
   - X_test.npy,  y_test.npy
   - metadata.json

Recommended default label rule
------------------------------
Window label = attack class if ANY frame in the window is attack, else 0.
This is usually a good starting point for IDS window classification.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


@dataclass
class PreprocessConfig:
    input_dir: str
    output_dir: str
    window_size: int = 64
    train_stride: int = 4
    eval_stride: int = 1

    train_ratio: float = 0.70
    val_ratio: float = 0.10
    test_ratio: float = 0.20

    # Guard band removes a small region around split boundaries to reduce leakage
    guard_frames: int = 64

    # Frequency feature: count same ID in the last `freq_history` frames (including current)
    freq_history: int = 128

    # Minimum frames needed per file to be usable
    min_frames_per_file: int = 1000

    # Save float dtype
    dtype: str = "float32"

    # Labeling strategy for a window
    # "any_attack" -> class_id if any attack frame appears in the window, else 0
    # "last_frame" -> label by the last frame
    label_rule: str = "any_attack"


def safe_hex_to_bytes(hex_str: str, max_len: int = 8) -> Tuple[np.ndarray, int]:
    """
    Convert hex payload string to 8 payload bytes + DLC.
    Examples:
        "0200000000000002" -> [2,0,0,0,0,0,0,2], dlc=8
        "A1B2" -> [161,178,0,0,0,0,0,0], dlc=2
    """
    if pd.isna(hex_str):
        return np.zeros(max_len, dtype=np.float32), 0

    s = str(hex_str).strip().lower().replace(" ", "")
    if s.startswith("0x"):
        s = s[2:]

    # if odd number of chars, pad left with zero
    if len(s) % 2 == 1:
        s = "0" + s

    # truncate to at most 8 bytes = 16 hex chars
    s = s[: max_len * 2]

    byte_vals = []
    for i in range(0, len(s), 2):
        try:
            byte_vals.append(int(s[i:i+2], 16))
        except ValueError:
            byte_vals.append(0)

    dlc = len(byte_vals)
    if dlc < max_len:
        byte_vals.extend([0] * (max_len - dlc))

    return np.array(byte_vals, dtype=np.float32), dlc


def add_basic_features(df: pd.DataFrame, freq_history: int) -> pd.DataFrame:
    """
    Add raw per-frame features before scaling:
    - payload_0 ... payload_7
    - dlc
    - delta_t
    - log_delta_t
    - id_freq_recent
    """
    df = df.copy()
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    # Arbitration ID as integer
    df["arbitration_id"] = pd.to_numeric(df["arbitration_id"], errors="coerce").fillna(0).astype(np.int64)

    # Attack column
    df["attack"] = pd.to_numeric(df["attack"], errors="coerce").fillna(0).astype(np.int64)

    # Payload bytes and DLC
    payload_arr = np.zeros((len(df), 8), dtype=np.float32)
    dlc_arr = np.zeros(len(df), dtype=np.int64)

    for i, s in enumerate(df["data_field"].values):
        payload, dlc = safe_hex_to_bytes(s, max_len=8)
        payload_arr[i] = payload
        dlc_arr[i] = dlc

    for j in range(8):
        df[f"payload_{j}"] = payload_arr[:, j]
    df["dlc"] = dlc_arr

    # Delta time
    ts = pd.to_numeric(df["timestamp"], errors="coerce").fillna(method="ffill").fillna(method="bfill").values.astype(np.float64)
    delta_t = np.zeros(len(df), dtype=np.float64)
    if len(df) > 1:
        delta_t[1:] = np.maximum(ts[1:] - ts[:-1], 0.0)
    df["delta_t"] = delta_t
    df["log_delta_t"] = np.log1p(delta_t)

    # Local ID frequency in recent frame history
    ids = df["arbitration_id"].values
    freq_recent = np.zeros(len(df), dtype=np.float32)
    recent_counts: Dict[int, int] = {}
    recent_queue: List[int] = []

    for i, can_id in enumerate(ids):
        recent_queue.append(int(can_id))
        recent_counts[int(can_id)] = recent_counts.get(int(can_id), 0) + 1

        if len(recent_queue) > freq_history:
            old = recent_queue.pop(0)
            recent_counts[old] -= 1
            if recent_counts[old] == 0:
                del recent_counts[old]

        freq_recent[i] = recent_counts.get(int(can_id), 0)

    df["id_freq_recent"] = freq_recent
    return df


def split_indices(n: int, train_ratio: float, val_ratio: float, test_ratio: float, guard_frames: int):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, "Ratios must sum to 1"

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_start = 0
    val_start = train_end + guard_frames
    test_start = val_end + guard_frames

    train_end_adj = max(train_start, train_end - guard_frames)
    val_end_adj = max(val_start, val_end - guard_frames)

    train_idx = np.arange(train_start, train_end_adj, dtype=np.int64)
    val_idx = np.arange(val_start, val_end_adj, dtype=np.int64)
    test_idx = np.arange(test_start, n, dtype=np.int64)

    return train_idx, val_idx, test_idx


def build_scalers(train_frames: pd.DataFrame) -> Dict[str, MinMaxScaler]:
    scalers = {}

    scalers["arbitration_id"] = MinMaxScaler()
    scalers["dlc"] = MinMaxScaler()
    scalers["log_delta_t"] = MinMaxScaler()
    scalers["id_freq_recent"] = MinMaxScaler()

    scalers["arbitration_id"].fit(train_frames[["arbitration_id"]].values.astype(np.float32))
    scalers["dlc"].fit(train_frames[["dlc"]].values.astype(np.float32))
    scalers["log_delta_t"].fit(train_frames[["log_delta_t"]].values.astype(np.float32))
    scalers["id_freq_recent"].fit(train_frames[["id_freq_recent"]].values.astype(np.float32))
    return scalers


def transform_frames(df: pd.DataFrame, scalers: Dict[str, MinMaxScaler]) -> np.ndarray:
    """
    Final per-frame feature vector:
        [id_norm, dlc_norm, payload_0..7/255, log_delta_t_norm, id_freq_recent_norm]
    -> total dim = 12
    """
    id_norm = scalers["arbitration_id"].transform(df[["arbitration_id"]].values.astype(np.float32))
    dlc_norm = scalers["dlc"].transform(df[["dlc"]].values.astype(np.float32))

    payload = df[[f"payload_{j}" for j in range(8)]].values.astype(np.float32) / 255.0

    log_dt_norm = scalers["log_delta_t"].transform(df[["log_delta_t"]].values.astype(np.float32))
    freq_norm = scalers["id_freq_recent"].transform(df[["id_freq_recent"]].values.astype(np.float32))

    features = np.concatenate(
        [id_norm, dlc_norm, payload, log_dt_norm, freq_norm],
        axis=1
    ).astype(np.float32)

    return features


def make_windows(
    features: np.ndarray,
    attack_binary: np.ndarray,
    class_id: int,
    window_size: int,
    stride: int,
    label_rule: str = "any_attack",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    features: (N, C)
    attack_binary: (N,) with values 0/1
    class_id: file-specific attack class (1..9)
    """
    xs = []
    ys = []

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

    X = np.stack(xs, axis=0).astype(np.float32) if xs else np.empty((0, window_size, features.shape[1]), dtype=np.float32)
    y = np.asarray(ys, dtype=np.int64)
    return X, y


def process_one_file(
    csv_path: Path,
    class_id: int,
    cfg: PreprocessConfig,
) -> Dict[str, pd.DataFrame]:
    df = pd.read_csv(csv_path)

    required_cols = {"timestamp", "arbitration_id", "data_field", "attack"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} missing required columns: {missing}")

    df = add_basic_features(df, freq_history=cfg.freq_history)

    if len(df) < cfg.min_frames_per_file:
        raise ValueError(
            f"{csv_path.name} has only {len(df)} frames, smaller than min_frames_per_file={cfg.min_frames_per_file}"
        )

    train_idx, val_idx, test_idx = split_indices(
        len(df),
        cfg.train_ratio,
        cfg.val_ratio,
        cfg.test_ratio,
        cfg.guard_frames,
    )

    return {
        "train": df.iloc[train_idx].reset_index(drop=True),
        "val": df.iloc[val_idx].reset_index(drop=True),
        "test": df.iloc[test_idx].reset_index(drop=True),
        "file_name": csv_path.name,
        "class_id": class_id,
    }


def save_metadata(
    out_dir: Path,
    cfg: PreprocessConfig,
    class_map: Dict[str, int],
    feature_dim: int,
    stats: Dict,
):
    meta = {
        "config": cfg.__dict__,
        "class_map": class_map,
        "feature_description": [
            "arbitration_id_norm",
            "dlc_norm",
            "payload_0_norm255",
            "payload_1_norm255",
            "payload_2_norm255",
            "payload_3_norm255",
            "payload_4_norm255",
            "payload_5_norm255",
            "payload_6_norm255",
            "payload_7_norm255",
            "log_delta_t_norm",
            "id_freq_recent_norm",
        ],
        "feature_dim": feature_dim,
        "stats": stats,
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def preprocess_can_multiclass(cfg: PreprocessConfig):
    input_dir = Path(cfg.input_dir)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.glob("*.csv"))
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    print(f"Found {len(csv_files)} CSV files.")

    # Map each file to one attack class: 1..N
    class_map = {csv_path.name: i + 1 for i, csv_path in enumerate(csv_files)}

    # Process each file first
    per_file = []
    for csv_path in csv_files:
        class_id = class_map[csv_path.name]
        print(f"Processing {csv_path.name} -> attack class {class_id}")
        data = process_one_file(csv_path, class_id, cfg)
        per_file.append(data)

    # Fit scalers using ALL training frames across all files
    train_frames_all = pd.concat([d["train"] for d in per_file], axis=0, ignore_index=True)
    scalers = build_scalers(train_frames_all)

    # Build windows for each split
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    X_test_list, y_test_list = [], []

    stats = {
        "files": [],
        "num_classes_total": len(csv_files) + 1,  # 0 = normal, 1..N = attacks
    }

    for data in per_file:
        file_name = data["file_name"]
        class_id = data["class_id"]

        file_stat = {"file_name": file_name, "class_id": class_id}

        for split_name, stride, x_list, y_list in [
            ("train", cfg.train_stride, X_train_list, y_train_list),
            ("val", cfg.eval_stride, X_val_list, y_val_list),
            ("test", cfg.eval_stride, X_test_list, y_test_list),
        ]:
            df_split = data[split_name]
            feats = transform_frames(df_split, scalers)
            attack_bin = df_split["attack"].values.astype(np.int64)

            X_split, y_split = make_windows(
                features=feats,
                attack_binary=attack_bin,
                class_id=class_id,
                window_size=cfg.window_size,
                stride=stride,
                label_rule=cfg.label_rule,
            )

            x_list.append(X_split)
            y_list.append(y_split)

            file_stat[f"{split_name}_frames"] = int(len(df_split))
            file_stat[f"{split_name}_windows"] = int(len(X_split))
            file_stat[f"{split_name}_normal_windows"] = int(np.sum(y_split == 0))
            file_stat[f"{split_name}_attack_windows"] = int(np.sum(y_split == class_id))

        stats["files"].append(file_stat)

    # Concatenate all files
    X_train = np.concatenate(X_train_list, axis=0).astype(cfg.dtype) if X_train_list else np.empty((0, cfg.window_size, 12), dtype=cfg.dtype)
    y_train = np.concatenate(y_train_list, axis=0).astype(np.int64) if y_train_list else np.empty((0,), dtype=np.int64)

    X_val = np.concatenate(X_val_list, axis=0).astype(cfg.dtype) if X_val_list else np.empty((0, cfg.window_size, 12), dtype=cfg.dtype)
    y_val = np.concatenate(y_val_list, axis=0).astype(np.int64) if y_val_list else np.empty((0,), dtype=np.int64)

    X_test = np.concatenate(X_test_list, axis=0).astype(cfg.dtype) if X_test_list else np.empty((0, cfg.window_size, 12), dtype=cfg.dtype)
    y_test = np.concatenate(y_test_list, axis=0).astype(np.int64) if y_test_list else np.empty((0,), dtype=np.int64)

    # Shuffle ONLY training set
    if len(X_train) > 0:
        perm = np.random.permutation(len(X_train))
        X_train = X_train[perm]
        y_train = y_train[perm]

    # Save arrays
    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "X_val.npy", X_val)
    np.save(out_dir / "y_val.npy", y_val)
    np.save(out_dir / "X_test.npy", X_test)
    np.save(out_dir / "y_test.npy", y_test)

    save_metadata(
        out_dir=out_dir,
        cfg=cfg,
        class_map=class_map,
        feature_dim=X_train.shape[-1] if len(X_train) > 0 else 12,
        stats=stats,
    )

    print("\nSaved:")
    print(f"  {out_dir / 'X_train.npy'}  {X_train.shape}")
    print(f"  {out_dir / 'y_train.npy'}  {y_train.shape}")
    print(f"  {out_dir / 'X_val.npy'}    {X_val.shape}")
    print(f"  {out_dir / 'y_val.npy'}    {y_val.shape}")
    print(f"  {out_dir / 'X_test.npy'}   {X_test.shape}")
    print(f"  {out_dir / 'y_test.npy'}   {y_test.shape}")
    print(f"  {out_dir / 'metadata.json'}")

    # Print label distribution
    def show_dist(name, y):
        if len(y) == 0:
            print(f"{name}: empty")
            return
        vals, cnts = np.unique(y, return_counts=True)
        print(f"{name} label distribution:")
        for v, c in zip(vals, cnts):
            print(f"  class {int(v):2d}: {int(c)}")

    print()
    show_dist("Train", y_train)
    show_dist("Val", y_val)
    show_dist("Test", y_test)


if __name__ == "__main__":
    # Example usage:
    # Update these paths before running.
    cfg = PreprocessConfig(
        input_dir="./data/2017-subaru-forester/merged",
        output_dir="./data/2017-subaru-forester/preprocessed",
        window_size=64,
        train_stride=4,
        eval_stride=1,
        train_ratio=0.70,
        val_ratio=0.10,
        test_ratio=0.20,
        guard_frames=64,
        freq_history=128,
        min_frames_per_file=1000,
        label_rule="any_attack",
    )

    preprocess_can_multiclass(cfg)
