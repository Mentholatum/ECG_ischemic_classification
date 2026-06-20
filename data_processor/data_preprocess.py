#!/usr/bin/env python3
# convert_to_h5_clean.py
# --------------------------------------------------------
# 纯 WFDB -> HDF5 转换（清洗 NaN 版）
# 功能：跳过含 NaN 的样本，同时生成干净的 labels_clean.csv
# --------------------------------------------------------

import os
import pandas as pd
import numpy as np
import wfdb
import h5py
from scipy.signal import resample
from tqdm import tqdm

BASE_DIR = "/media/ssd/jiachuang/data/medical/Heart/301/ecg-diagnostic-electrocardiogram-matched-subset"
CSV_PATH = '/media/ssd/jiachuang/data/medical/Heart/301/ecg_hfref_relative.csv'
OUTPUT_DIR = os.path.join(BASE_DIR, "h5_format")
H5_PATH = os.path.join(OUTPUT_DIR, "ecg_data_clean.h5")

TARGET_FS = 200
TARGET_LEN = 2000
N_LEADS = 12

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_csv_robust(path):
    with open(path, 'r') as f:
        first = f.readline()
    sep = '\t' if '\t' in first else ','
    df = pd.read_csv(path, sep=sep, engine='python', on_bad_lines='skip')
    if len(df.columns) == 1 and sep in str(df.columns[0]):
        col_names = first.strip().split(sep)
        df = pd.read_csv(path, sep=sep, engine='python', names=col_names, skiprows=1, on_bad_lines='skip')
    return df


def preprocess_wfdb(record_path, target_fs=200, target_len=2000, n_leads=12):
    record = wfdb.rdrecord(record_path)
    sig = record.p_signal
    fs = record.fs

    if sig is None or sig.shape[1] == 0:
        raise ValueError("Empty signal")
    if sig.shape[1] < n_leads:
        raise ValueError(f"Only {sig.shape[1]} leads found, expected {n_leads}")

    # 检查原始信号是否含 NaN
    if np.isnan(sig).any():
        raise ValueError("Raw signal contains NaN")

    sig = sig[:, :n_leads].T.astype(np.float64)

    if fs != target_fs:
        n_samples = int(sig.shape[1] * target_fs / fs)
        sig = resample(sig, n_samples, axis=1)

    if sig.shape[1] > target_len:
        sig = sig[:, :target_len]
    elif sig.shape[1] < target_len:
        pad = target_len - sig.shape[1]
        sig = np.pad(sig, ((0, 0), (0, pad)), mode='constant')

    sig = sig / 100.0
    return sig.astype(np.float32)


def main():
    df = load_csv_robust(CSV_PATH)
    print(f"Loaded CSV: {len(df)} rows")

    # 提取三列，用于后续生成干净标签
    label_df = df[['study_id', 'subject_id', 'is_ischemic']].copy()
    label_df['study_id'] = label_df['study_id'].astype(int)
    label_df['subject_id'] = label_df['subject_id'].astype(int)
    label_df['is_ischemic'] = label_df['is_ischemic'].astype(int)

    failed = []
    nan_records = []
    success = 0
    clean_indices = []

    with h5py.File(H5_PATH, 'w') as h5f:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting"):
            rel_path = str(row.get('ecg_file_path', '')).strip()
            if not rel_path:
                npy_path = str(row.get('npy_path', '')).strip()
                rel_path = npy_path.replace('.npy', '') if npy_path else ''

            if not rel_path:
                failed.append((idx, "Missing path"))
                continue

            full_path = os.path.join(BASE_DIR, rel_path)
            if not os.path.exists(full_path + ".hea") and not os.path.exists(full_path + ".dat"):
                failed.append((idx, f"Not found: {full_path}"))
                continue

            try:
                sig = preprocess_wfdb(full_path, TARGET_FS, TARGET_LEN, N_LEADS)
                sid = str(int(row['study_id']))

                h5f.create_dataset(
                    sid,
                    data=sig,
                    dtype='float32',
                    compression='gzip',
                    compression_opts=4,
                    chunks=(12, 2000),
                )
                success += 1
                clean_indices.append(idx)

            except ValueError as e:
                if "NaN" in str(e):
                    nan_records.append((sid if 'sid' in dir() else str(row['study_id']), str(e)))
                else:
                    failed.append((idx, f"{rel_path}: {e}"))
            except Exception as e:
                failed.append((idx, f"{rel_path}: {e}"))
                continue

    # 生成干净的 labels_clean.csv（只包含成功写入 HDF5 的样本）
    clean_df = label_df.iloc[clean_indices].copy()
    clean_csv = os.path.join(OUTPUT_DIR, "labels_clean.csv")
    clean_df.to_csv(clean_csv, index=False)

    print(f"\n{'='*50}")
    print(f"HDF5: {H5_PATH}")
    h5_size = os.path.getsize(H5_PATH)
    print(f"Size: {h5_size / 1024 / 1024:.1f} MB")
    print(f"Success: {success} / {len(df)}")
    print(f"NaN filtered: {len(nan_records)}")
    print(f"Other failed: {len(failed)}")
    print(f"Clean CSV: {clean_csv} ({len(clean_df)} rows)")

    if nan_records:
        nan_log = os.path.join(OUTPUT_DIR, "nan_records.txt")
        with open(nan_log, 'w') as f:
            for sid, reason in nan_records:
                f.write(f"{sid}: {reason}\n")
        print(f"NaN log: {nan_log}")

    if failed:
        fail_log = os.path.join(OUTPUT_DIR, "failed.txt")
        with open(fail_log, 'w') as f:
            for idx, reason in failed:
                f.write(f"{idx}: {reason}\n")
        print(f"Fail log: {fail_log}")

    print('='*50)


if __name__ == "__main__":
    main()