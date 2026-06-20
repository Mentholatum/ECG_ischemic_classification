#!/usr/bin/env python3
# train_fromscratch.py
# --------------------------------------------------------
# 12-lead ECG 二分类 from scratch 训练
# --------------------------------------------------------

import os
import json
import argparse
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score,
    balanced_accuracy_score, average_precision_score,
    confusion_matrix,
)

from ecgnet import ECGNet
from data_processor.dataset import get_ecg_dataset


# ========================== 0. 工具类 ==========================
class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    @property
    def global_avg(self):
        return self.sum / max(self.count, 1)


class MetricLogger:
    def __init__(self, delimiter="  "):
        self.delimiter = delimiter
        self.meters = {}

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if k not in self.meters:
                self.meters[k] = AverageMeter()
            self.meters[k].update(v)

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if header:
            print(header)
        for obj in iterable:
            yield obj
            i += 1
            if i % print_freq == 0 and self.meters:
                stats = []
                for k, meter in self.meters.items():
                    stats.append(f"{k}: {meter.global_avg:.4f}")
                print(f"[{i:04d}] " + self.delimiter.join(stats))


class ECGAugment(nn.Module):
    def __init__(self, noise_std=0.02, scale_range=(0.9, 1.1), p=0.3):
        super().__init__()
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.p = p

    def forward(self, x):
        # x: (B, 12, 2000)

        # 1. 高斯噪声
        if torch.rand(1).item() < self.p:
            x = x + torch.randn_like(x) * self.noise_std

        # 2. 幅度缩放
        if torch.rand(1).item() < self.p:
            scale = torch.empty(x.size(0), 1, 1).uniform_(*self.scale_range).to(x.device)
            x = x * scale

        # 3. 循环时移（已有，保留）
        if torch.rand(1).item() < self.p:
            shift = torch.randint(0, x.shape[2], (1,)).item()
            x = torch.roll(x, shift, dims=2)

        # 4. 基线漂移（模拟呼吸/电极移动）
        if torch.rand(1).item() < self.p:
            t = torch.linspace(0, 4 * 3.14159, x.shape[2], device=x.device)
            phase = torch.rand(x.size(0), 1, 1, device=x.device) * 2 * 3.14159
            drift = 0.05 * torch.sin(t + phase)
            x = x + drift

        # 5. 频域随机掩码（模拟肌电噪声）
        if torch.rand(1).item() < self.p:
            x_fft = torch.fft.rfft(x, dim=2)
            mask = torch.rand_like(x_fft.real) > 0.1  # 随机丢弃10%频域成分
            x_fft = x_fft * mask
            x = torch.fft.irfft(x_fft, n=x.shape[2], dim=2)

        # 6. 统一时间拉伸（整 batch 相同 scale，避免 item() 报错）
        if torch.rand(1).item() < self.p:
            scale = np.random.uniform(*self.scale_range)  # 标量
            x = torch.nn.functional.interpolate(
                x, scale_factor=scale, mode='linear', align_corners=False
            )
            # 裁剪/填充回 2000
            if x.shape[2] > 2000:
                x = x[:, :, :2000]
            elif x.shape[2] < 2000:
                pad = 2000 - x.shape[2]
                x = torch.nn.functional.pad(x, (0, pad), mode='replicate')

        return x


class FocalLoss(nn.Module):
    """
    Focal Loss for 70% positive rate:
    - alpha=0.15: 正例(多数类)权重低，负例(少数类)权重高(0.85)
    - gamma=2.0:  压制易分样本梯度
    """
    def __init__(self, alpha=0.15, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        # targets=1(阳性) -> alpha=0.15; targets=0(阴性) -> 1-alpha=0.85
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        return loss.mean()

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        """
        alpha: FP 权重 (控制 Spec)
        beta:  FN 权重 (控制 Sens)
        当前 Spec 低 -> FP 多 -> 增大 alpha 惩罚 FP
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        TP = (probs * targets).sum()
        FP = (probs * (1 - targets)).sum()
        FN = ((1 - probs) * targets).sum()
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return 1 - tversky


# ========================== 1. 损失函数 ==========================
def get_criterion(train_labels, device):
    # 1:1 平衡数据，无需任何加权，标准 BCE 最稳定
    return nn.BCEWithLogitsLoss().to(device)


# ========================== 2. 预处理 ==========================
def normalize_ecg(x):
    """
    逐样本、逐通道 z-score 标准化
    dataset.py 已去掉 x*100，此处直接标准化原始幅度
    """
    mean = x.mean(dim=2, keepdim=True)
    std = x.std(dim=2, keepdim=True) + 1e-6
    return (x - mean) / std


# ========================== 3. 训练与评估 ==========================
def train_one_epoch(model, criterion, train_loader, optimizer, device, scaler, augment=None):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")

    for step, (samples, targets) in enumerate(metric_logger.log_every(train_loader, 10, 'Train')):
        samples = samples.float().to(device, non_blocking=True)
        samples = normalize_ecg(samples)
        if augment is not None:
            samples = augment(samples)

        targets = targets.to(device, non_blocking=True).float().unsqueeze(-1)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        scaler.step(optimizer)
        scaler.update()

        metric_logger.update(loss=loss.item())

    return metric_logger.meters['loss'].global_avg


@torch.no_grad()
def evaluate(model, data_loader, device, header='Test:'):
    model.eval()
    all_preds = []
    all_targets = []

    for samples, targets in tqdm(data_loader, desc=header, leave=False):
        samples = samples.float().to(device, non_blocking=True)
        samples = normalize_ecg(samples)

        targets = targets.to(device, non_blocking=True).float().unsqueeze(-1)

        with torch.cuda.amp.autocast():
            outputs = model(samples)

        probs = torch.sigmoid(outputs)
        all_preds.append(probs.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0).squeeze()
    targets = np.concatenate(all_targets, axis=0).squeeze()
    preds_bin = (preds > 0.5).astype(int)

    try:
        roc_auc = roc_auc_score(targets, preds)
    except ValueError:
        roc_auc = 0.0
    try:
        pr_auc = average_precision_score(targets, preds)
    except ValueError:
        pr_auc = 0.0

    f1 = f1_score(targets, preds_bin, zero_division=0)
    acc = accuracy_score(targets, preds_bin)
    bal_acc = balanced_accuracy_score(targets, preds_bin)

    cm = confusion_matrix(targets, preds_bin).ravel()
    if len(cm) == 4:
        tn, fp, fn, tp = cm
    else:
        if targets.sum() == 0:
            tn, fp, fn, tp = len(targets), 0, 0, 0
        else:
            tn, fp, fn, tp = 0, 0, 0, len(targets)

    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)

    metrics = {
        'f1': float(f1), 'roc_auc': float(roc_auc), 'accuracy': float(acc),
        'balanced_accuracy': float(bal_acc), 'pr_auc': float(pr_auc),
        'sensitivity': float(sens), 'specificity': float(spec),
    }

    print(f"{header} F1={f1:.4f} | AUC={roc_auc:.4f} | Acc={acc:.4f} | "
          f"BalAcc={bal_acc:.4f} | Sens={sens:.4f} | Spec={spec:.4f}")
    return metrics


# ========================== 4. 核心训练 ==========================
def format_time(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))

def train_fold(model, train_loader, val_loader, test_loader, args, device, fold_idx):

    # 1. 损失（默认 Focal Loss）
    train_labels = [label for _, label in train_loader.dataset]
    criterion = get_criterion(
        train_labels, device,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"[Fold {fold_idx}] Trainable params: {n_trainable / 1e6:.3f}M")

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    scaler = torch.cuda.amp.GradScaler()

    best_score = -1.0
    best_epoch = -1
    best_metrics = None
    epochs_no_improve = 0
    patience = 150

    output_dir = Path(args.output_dir) / f"fold_{fold_idx}"
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_log_path = output_dir / "eval_log.txt"
    with open(eval_log_path, 'w') as f:
        f.write("Epoch | TrainLoss | F1 | AUC | Acc | BalAcc | Sens | Spec | Score | Status \n")

    augment = ECGAugment() if args.augment else None

    for epoch in range(args.epochs):

        train_loss = train_one_epoch(model, criterion, train_loader, optimizer, device, scaler, augment)
        val_metrics = evaluate(model, val_loader, device, header='Val:')

        val_f1 = val_metrics['f1']
        val_auc = val_metrics['roc_auc']
        val_balacc = val_metrics['balanced_accuracy']
        val_spec = val_metrics['specificity']
        val_sens = val_metrics['sensitivity']
        score = val_auc + val_balacc + 0.3 * min(val_spec, val_sens)


        status = "  "
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_metrics = val_metrics
            epochs_no_improve = 0

            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'best_score': best_score,
                'best_f1': val_f1,
                'best_auc': val_auc,
                'val_metrics': val_metrics,
            }, output_dir / "best_model.pth")
            status = "-> BEST"
            print(f"  >>> New best Score={best_score:.4f} (F1={val_f1:.4f}, AUC={val_auc:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  -> No improve: {epochs_no_improve}/{patience} (best Score={best_score:.4f} @ {best_epoch})")

        # 写入日志
        with open(eval_log_path, 'a') as f:
            f.write(f"{epoch:03d} | {train_loss:.6f} | {val_f1:.4f} | {val_auc:.4f} | "
                    f"{val_metrics['accuracy']:.4f} | {val_metrics['balanced_accuracy']:.4f} | "
                    f"{val_metrics['sensitivity']:.4f} | {val_metrics['specificity']:.4f} | "
                    f"{score:.4f} | {status}\n")

        # 打印带 Epoch的状态
        print(f"[Fold {fold_idx}] | Loss: {train_loss:.4f} | "
              f"F1={val_f1:.4f} AUC={val_auc:.4f} Score={score:.4f} | {status}")

        if epochs_no_improve >= patience:
            print(f"\n[Fold {fold_idx}] Early stop @ epoch {epoch}. "
                  f"Best Score={best_score:.4f} @ {best_epoch}")
            break

        scheduler.step()

    # 5. 加载最优模型，最终测试
    print(f"\n[Fold {fold_idx}] Loading best model from epoch {best_epoch} for final test...")
    checkpoint = torch.load(output_dir / "best_model.pth", map_location=device)
    model.load_state_dict(checkpoint['model'])

    test_metrics = evaluate(model, test_loader, device, header='Test:')

    with open(eval_log_path, 'a') as f:
        f.write(f"{'=' * 70}\n")
        f.write(f"BEST | Epoch {best_epoch} | Score={best_score:.4f} | "
                f"F1={best_metrics['f1']:.4f} | AUC={best_metrics['roc_auc']:.4f}\n")
        f.write(f"TEST | - | - | F1={test_metrics['f1']:.4f} | AUC={test_metrics['roc_auc']:.4f} | "
                f"Acc={test_metrics['accuracy']:.4f} | BalAcc={test_metrics['balanced_accuracy']:.4f} | FINAL\n")

    with open(output_dir / "test_metrics.json", 'w') as f:
        json.dump({
            'best_epoch': best_epoch,
            'best_score': best_score,
            'best_val_metrics': best_metrics,
            'test_metrics': test_metrics,
        }, f, indent=2)

    return test_metrics


# ========================== 5. 参数与主入口 ==========================
def get_args():
    parser = argparse.ArgumentParser('ECG 5-Fold CV (From Scratch)')
    parser.add_argument('--output_dir', default='./ecg_crnn_output', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=600, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--drop', default=0.1, type=float, help='dropout for GRU & classifier')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--seed', default=2026, type=int)

    # 数据增强
    parser.add_argument('--augment', action='store_true', default=True)

    return parser.parse_args()


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    data_root = "/media/ssd/jiachuang/data/medical/Heart/301/ecg-diagnostic-electrocardiogram-matched-subset/h5_format"

    all_results = []

    for fold_idx in range(5):
        print(f"\n{'=' * 70}")
        print(f"  Fold {fold_idx}/5 | Train=others, Val=Test=Fold {fold_idx}")
        print(f"{'=' * 70}")

        train_dataset, test_dataset, _, _, _ = get_ecg_dataset(data_root, fold_idx)
        val_dataset = test_dataset

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size * 2, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, drop_last=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size * 2, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, drop_last=False
        )

        model = ECGNet(in_channels=12, n_classes=1, dropout=args.drop).to(device)
        print(f"[DEBUG] Last layer bias: {model.classifier[-1].bias.item()}")
        print(f"[DEBUG] Last layer weight mean: {model.classifier[-1].weight.mean().item()}")
        total, trainable = model.get_trainable_params()
        print(f"Model: Total={total/1e6:.3f}M, Trainable={trainable/1e6:.3f}M")

        result = train_fold(
            model, train_loader, val_loader, test_loader,
            args, device, fold_idx
        )
        all_results.append(result)

    # 汇总
    print(f"\n{'=' * 70}")
    print("  5-Fold Summary")
    print(f"{'=' * 70}")
    for i, r in enumerate(all_results):
        print(f"  Fold {i}: F1={r['f1']:.4f} | AUC={r['roc_auc']:.4f} | "
              f"Acc={r['accuracy']:.4f} | BalAcc={r['balanced_accuracy']:.4f} | "
              f"Sens={r['sensitivity']:.4f} | Spec={r['specificity']:.4f}")

    summary = {}
    for key in ['f1', 'roc_auc', 'accuracy', 'balanced_accuracy', 'pr_auc', 'sensitivity', 'specificity']:
        vals = [r.get(key, 0.0) for r in all_results]
        summary[f"{key}_mean"] = float(np.mean(vals))
        summary[f"{key}_std"] = float(np.std(vals))

    print(f"\n  Mean±Std: F1={summary['f1_mean']:.4f}±{summary['f1_std']:.4f} | "
          f"AUC={summary['roc_auc_mean']:.4f}±{summary['roc_auc_std']:.4f} | "
          f"BalAcc={summary['balanced_accuracy_mean']:.4f}±{summary['balanced_accuracy_std']:.4f} | "
          f"Spec={summary['specificity_mean']:.4f}±{summary['specificity_std']:.4f}")

    summary_path = Path(args.output_dir) / "cv_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({'per_fold': all_results, 'summary': summary, 'config': vars(args)}, f, indent=2)
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()