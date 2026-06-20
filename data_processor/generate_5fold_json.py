#!/usr/bin/env python3
# generate_5fold_json_balanced.py
# --------------------------------------------------------
# 构建 1:1 平衡子集 + 标准五折分层分组交叉验证
# 策略：以阴性为基准，随机采样等量阳性 subject，然后 StratifiedGroupKFold
# --------------------------------------------------------

import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

LABELS_CSV = "/media/ssd/jiachuang/data/medical/Heart/301/ecg-diagnostic-electrocardiogram-matched-subset/h5_format/labels_clean.csv"
OUTPUT_DIR = "/media/ssd/jiachuang/data/medical/Heart/301/ecg-diagnostic-electrocardiogram-matched-subset/h5_format"
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 读取干净标签
df = pd.read_csv(LABELS_CSV)
print(f"Original clean samples: {len(df)}")
print(f"Original subjects: {df['subject_id'].nunique()}")
print(f"Original class distribution:\n{df['is_ischemic'].value_counts()}\n")

# 按 subject 聚合
subject_df = (
    df.groupby('subject_id')
    .agg(
        label=('is_ischemic', lambda x: int(x.mode()[0])),
        n_samples=('study_id', 'count'),
        study_ids=('study_id', lambda x: x.astype(str).tolist())
    )
    .reset_index()
)

pos_subjects = subject_df[subject_df['label'] == 1].copy()
neg_subjects = subject_df[subject_df['label'] == 0].copy()

n_pos_total = pos_subjects['n_samples'].sum()
n_neg_total = neg_subjects['n_samples'].sum()
print(f"Positive: {len(pos_subjects)} subjects, {n_pos_total} samples")
print(f"Negative: {len(neg_subjects)} subjects, {n_neg_total} samples")

# --------------------------------------------------------
# 1. 构建 1:1 平衡子集：从阳性中随机采样，匹配阴性样本总量
# --------------------------------------------------------
target_samples = n_neg_total  # 以阴性为基准

# 随机打乱阳性 subjects，按累积样本数取最接近 target 的一批
pos_shuffled = pos_subjects.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
cumsum = pos_shuffled['n_samples'].cumsum().values

# 找到使累积样本数最接近 target 的切分索引
best_idx = np.argmin(np.abs(cumsum - target_samples))
selected_pos = pos_shuffled.iloc[:best_idx + 1].copy()

n_selected_pos = selected_pos['n_samples'].sum()
print(f"\n[Balanced] Selected {len(selected_pos)} positive subjects ({n_selected_pos} samples)")
print(f"[Balanced] Kept   {len(neg_subjects)} negative subjects ({n_neg_total} samples)")
print(f"[Balanced] Ratio:  {n_selected_pos}:{n_neg_total} ≈ 1:{n_neg_total/n_selected_pos:.2f}")

# 合并平衡数据集
balanced_df = pd.concat([selected_pos, neg_subjects], ignore_index=True)
print(f"[Balanced] Total subjects: {len(balanced_df)} | Total samples: {balanced_df['n_samples'].sum()}\n")

# --------------------------------------------------------
# 2. 标准 StratifiedGroupKFold（5 折，val = test）
# --------------------------------------------------------
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
subjects = balanced_df['subject_id'].values
labels = balanced_df['label'].values

all_folds = {}

for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(subjects, labels, groups=subjects)):
    train_subjects = subjects[train_idx]
    test_subjects = subjects[test_idx]
    val_subjects = test_subjects  # val = test

    def collect(subject_list):
        sid_list = balanced_df[balanced_df['subject_id'].isin(subject_list)]['study_ids'].explode().tolist()
        sub_df = df[df['study_id'].astype(str).isin(sid_list)]
        return {
            "study_ids": sub_df['study_id'].astype(str).tolist(),
            "labels": sub_df['is_ischemic'].astype(int).tolist()
        }

    fold_data = {
        "train": collect(train_subjects),
        "val": collect(val_subjects),
        "test": collect(test_subjects)
    }
    all_folds[f"fold_{fold_idx}"] = fold_data

    print(f"Fold {fold_idx}:")
    for split_name in ['train', 'val', 'test']:
        n = len(fold_data[split_name]['labels'])
        n_pos = sum(fold_data[split_name]['labels'])
        ratio = n_pos / n if n > 0 else 0
        print(f"  {split_name:5s}: {n:6d} samples | pos={n_pos:5d} ({ratio*100:.1f}%) | neg={n-n_pos:5d} ({(1-ratio)*100:.1f}%)")

# 保存总 JSON
total_json = os.path.join(OUTPUT_DIR, "5fold_split_balanced.json")
with open(total_json, 'w') as f:
    json.dump(all_folds, f, indent=2)
print(f"\n[Saved] {total_json}")

# 保存单个 fold JSON
for fold_idx in range(5):
    single = {f"fold_{fold_idx}": all_folds[f"fold_{fold_idx}"]}
    with open(os.path.join(OUTPUT_DIR, f"fold_{fold_idx}_balanced.json"), 'w') as f:
        json.dump(single, f, indent=2)

print("Done.")