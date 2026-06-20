#!/usr/bin/env python3
# dataset.py

import os
import json
import h5py
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class ECGFoldDataset(Dataset):
    """
    五折 ECG Dataset（内存缓存版，数据已清洗无 NaN）
    """

    def __init__(self, h5_path, fold_json_path, fold_idx=0, split='train'):
        # 1. 加载索引
        with open(fold_json_path, 'r') as f:
            data = json.load(f)
        if f"fold_{fold_idx}" in data:
            split_data = data[f"fold_{fold_idx}"][split]
        else:
            split_data = data[split]

        self.study_ids = [str(s) for s in split_data['study_ids']]
        self.labels = [int(l) for l in split_data['labels']]

        # 2. 一次性加载当前 split 全部数据到内存
        n_samples = len(self.study_ids)
        print(f"[ECGFoldDataset-{split}] Loading {n_samples} samples into RAM cache...")

        self.data = {}
        with h5py.File(h5_path, 'r') as f:
            for sid in tqdm(self.study_ids, desc=f"Cache {split}", leave=False):
                self.data[sid] = f[sid][:]  # (12, 2000) numpy float32

        mem_gb = n_samples * 12 * 2000 * 4 / (1024 ** 3)
        print(f"[ECGFoldDataset-{split}] RAM cache ready. ~{mem_gb:.2f} GB")

    def __len__(self):
        return len(self.study_ids)

    def __getitem__(self, idx):
        sid = self.study_ids[idx]
        x = torch.from_numpy(self.data[sid]).float()  # 纯内存读取，零磁盘 I/O
        y = self.labels[idx]
        return x, y

def get_ecg_dataset(data_root, fold_idx=0):
    h5_path = os.path.join(data_root, "ecg_data_clean.h5")  # 新 HDF5
    json_path = os.path.join(data_root, "5fold_split_balanced.json")  # 新 JSON

    train = ECGFoldDataset(h5_path, json_path, fold_idx, 'train')
    val = ECGFoldDataset(h5_path, json_path, fold_idx, 'val')
    test = ECGFoldDataset(h5_path, json_path, fold_idx, 'test')

    ch_names = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8']
    metrics = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy", "f1"]

    return train, test, val, ch_names, metrics


# ========================== 多折循环训练辅助 ==========================
def get_ecg_dataset_fold(data_root, fold_idx):
    """
    如果你希望在一个脚本里循环训练 5 折，用这个函数切换 fold。
    """
    return get_ecg_dataset(data_root, fold_idx)