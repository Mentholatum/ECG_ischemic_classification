# ECG_ischemic_classification
> 基于多尺度时空特征与导联间关系建模的 12 导联心电图缺血性病变检测

## 项目简介

本项目针对 12 导联心电图（ECG）的缺血性病变检测任务，提出了一种融合 **多尺度卷积初始层**、**导联间关系自注意力** 与 **多尺度时序金字塔** 的深度网络架构 `ECGNet`。项目支持从原始 WFDB 格式数据到模型训练、五折交叉验证的完整流程，并内置了丰富的数据增强与早停机制，适用于高阳性率（~70%）的临床 ECG 数据集。

---

## 目录结构
```
ECG-Ischemia-Classification/
├── data_processor/
│   ├── data_preprocess.py      # 原始 WFDB -> HDF5 清洗与格式转换
│   ├── dataset.py              # PyTorch Dataset 与数据加载
│   └── generate_5fold_json.py  # 1:1 平衡五折交叉验证划分生成
├── ecgnet.py                # ECGNet 网络架构定义
├── train.py                    # 训练、验证与测试主脚本
├── requirements.txt            # Python 依赖
└── README.md                   # 本文件
```

---

## 1. 环境安装

本项目基于 Python 3.8+ 与 PyTorch 2.0 开发，支持 CUDA 11.8。请确保已安装 NVIDIA 驱动，并建议使用 Conda 或 venv 创建独立虚拟环境。

### 1.1 创建虚拟环境（推荐 Conda）

```bash
conda create -n ecg python=3.10 -y
conda activate ecg
```

### 1.2 安装依赖

项目依赖已整理于 `requirements.txt` 中，直接执行：

```bash
pip install -r requirements.txt
```

**`requirements.txt` 核心依赖说明：**

| 包 | 版本 | 用途 |
|---|---|---|
| `torch==2.0.1+cu118` | 2.0.1 | 深度学习框架（GPU 版） |
| `wfdb` | 4.3.1 | 读取原始 WFDB/PhysioNet ECG 数据 |
| `h5py` | 3.16.0 | 将清洗后的数据存储为 HDF5 格式，加速后续读取 |
| `scikit-learn` | 1.9.0 | 五折分层分组交叉验证（StratifiedGroupKFold） |
| `scipy` | 1.18.0 | 信号重采样 |
| `pandas` / `numpy` | 最新稳定版 | 数据清洗与数值计算 |
| `tqdm` | 4.68.2 | 进度条显示 |
| `tensorboardx` | 2.6.5 | 训练日志可视化（可选） |

> **注意**：若你的 CUDA 版本非 11.8，请前往 [PyTorch 官网](https://pytorch.org/get-started/previous-versions/) 选择对应版本的 `torch` 与 `torchvision` 命令手动安装，再安装其余依赖。

---

## 2. 数据预处理与划分

本项目采用 **WFDB -> HDF5 -> JSON 划分** 的两阶段数据准备流程，确保数据清洗、格式统一，并在患者级别（Subject-level）进行 1:1 平衡的五折交叉验证，避免同一患者的多条记录泄露到不同集合。

### 2.1 阶段一：原始数据清洗与格式转换 (`data_preprocess.py`)

**目的：**
- 读取原始 WFDB 格式（`.hea` + `.dat`）的 12 导联 ECG 信号；
- 将采样率统一重采样至 **200 Hz**，固定长度 **2000 点**（10 秒），仅保留 12 导联；
- 对信号幅度除以 100 进行初步缩放；
- **自动跳过包含 NaN 的异常样本**，保证数据质量；
- 将清洗后的信号写入 HDF5 文件（`gzip` 压缩），同时生成干净的标签文件 `labels_clean.csv`。

**运行前准备：**

请修改 `data_preprocess.py` 开头的路径变量，使其指向你的实际数据目录：

```python
BASE_DIR = "/media/ssd/jiachuang/data/medical/Heart/301/ecg-diagnostic-electrocardiogram-matched-subset"
CSV_PATH = '/media/ssd/jiachuang/data/medical/Heart/301/ecg_hfref_relative.csv'
OUTPUT_DIR = os.path.join(BASE_DIR, "h5_format")
```

其中 `CSV_PATH` 应包含至少以下三列：
- `study_id`：样本唯一编号
- `subject_id`：患者编号（用于后续分组）
- `is_ischemic`：标签（0 = 阴性，1 = 阳性）
- `ecg_file_path`：相对于 `BASE_DIR` 的 WFDB 记录路径（不含扩展名）

**运行命令：**

```bash
cd data_processor
python data_preprocess.py
```

**预期输出：**

```
HDF5: /.../h5_format/ecg_data_clean.h5
Size: xxx.x MB
Success: 89000 / 90000
NaN filtered: 500
Other failed: 500
Clean CSV: /.../h5_format/labels_clean.csv (89000 rows)
```

生成的 `ecg_data_clean.h5` 以 `study_id` 为 key，每个数据集形状为 `(12, 2000)`，数据类型 `float32`。

---

### 2.2 阶段二：构建 1:1 平衡五折划分 (`generate_5fold_json.py`)

**目的：**
- 在**患者级别（Subject-level）**进行阳性样本随机采样，使总阳性样本数与总阴性样本数接近 **1:1**；
- 采用 `StratifiedGroupKFold` 进行**五折分组交叉验证**，确保同一患者的所有记录只出现在同一个 Fold 中，彻底避免数据泄露；
- 生成 `5fold_split_balanced.json`，内含 `fold_0` 到 `fold_4` 的 `train` / `val` / `test` 索引（本项目采用 `val = test` 的独立测试策略）。

**运行前准备：**

确认 `LABELS_CSV` 和 `OUTPUT_DIR` 路径正确：

```python
LABELS_CSV = "/media/ssd/.../h5_format/labels_clean.csv"
OUTPUT_DIR = "/media/ssd/.../h5_format"
```

**运行命令：**

```bash
python generate_5fold_json.py
```

**预期输出：**

```
[Balanced] Selected 450 positive subjects (44500 samples)
[Balanced] Kept   500 negative subjects (44500 samples)
[Balanced] Ratio:  44500:44500 ≈ 1:1.00

Fold 0:
  train:  71200 samples | pos=35600 (50.0%) | neg=35600 (50.0%)
  val  :  17800 samples | pos= 8900 (50.0%) | neg= 8900 (50.0%)
  test :  17800 samples | pos= 8900 (50.0%) | neg= 8900 (50.0%)
...
[Saved] /.../h5_format/5fold_split_balanced.json
```

---

### 2.3 数据加载 (`dataset.py`)

`dataset.py` 中的 `ECGFoldDataset` 会在初始化时一次性将当前 Fold 对应 split 的全部数据从 HDF5 加载到内存中，后续训练时实现**零磁盘 I/O**，极大提升训练效率。

**使用方式（已在 `train.py` 中自动调用）：**

```python
from data_processor.dataset import get_ecg_dataset

train, test, val, ch_names, metrics = get_ecg_dataset(data_root, fold_idx=0)
```

---

## 3. 模型训练 (`train.py`)

### 3.1 训练流程概览

`train.py` 实现了完整的五折交叉验证训练流程，每折独立训练、验证与测试，并自动保存最优模型与汇总结果。

**核心特性：**
- **数据增强**：`ECGAugment` 模块集成高斯噪声、幅度缩放、循环时移、基线漂移、频域随机掩码、时间拉伸等 6 种增强策略；
- **损失函数**：默认使用 `BCEWithLogitsLoss`（平衡数据集下最稳定），代码中同时预置了 `FocalLoss` 与 `TverskyLoss` 接口，便于根据实际类别不平衡程度切换；
- **优化策略**：SGD（lr=1e-3, momentum=0.9, weight_decay=1e-4）+ `StepLR`（每 50 epoch 衰减 0.5 倍）；
- **混合精度训练**：`torch.cuda.amp` 自动加速；
- **早停机制**：`patience=150`，当验证综合得分连续 150 轮未提升则自动停止；
- **评估指标**：F1、ROC-AUC、PR-AUC、准确率、平衡准确率、敏感度（Sensitivity）、特异度（Specificity）。

**验证综合得分公式：**

```
Score = AUC + BalancedAccuracy + 0.3 * min(Specificity, Sensitivity)
```

该得分兼顾了排序能力（AUC）、整体判别能力（BalAcc）以及两类错误的均衡性（Sens/Spec）。

### 3.2 运行训练

**修改数据路径（如需要）：**

在 `train.py` 的 `main()` 函数中确认：

```python
data_root = "/media/ssd/jiachuang/data/medical/Heart/301/ecg-diagnostic-electrocardiogram-matched-subset/h5_format"
```

**启动五折交叉验证：**

```bash
python train.py
```

**关键参数说明（可通过命令行修改）：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--output_dir` | `./ecg_crnn_output` | 模型与日志保存目录 |
| `--batch_size` | 32 | 训练批次大小 |
| `--epochs` | 600 | 最大训练轮数 |
| `--lr` | 1e-3 | 初始学习率 |
| `--weight_decay` | 1e-4 | L2 正则化 |
| `--drop` | 0.1 | GRU 与分类头的 Dropout 率 |
| `--augment` | True | 是否开启数据增强 |

**示例（自定义参数）：**

```bash
python train.py --batch_size 64 --epochs 800 --lr 5e-4 --drop 0.2
```

### 3.3 训练日志与输出

每折训练结果保存在 `output_dir/fold_x/` 下：

```
fold_0/
├── best_model.pth          # 验证得分最优的模型权重
├── eval_log.txt            # 每轮训练/验证指标记录
└── test_metrics.json       # 最终测试集指标
```

五折全部结束后，根目录生成 `cv_summary.json`，汇总各折结果及 Mean±Std。

---

## 4. 网络架构介绍：ECGNet

`ECGNet` 是专为 12 导联 ECG 信号设计的深度分类网络，整体架构遵循 **"局部形态提取 → 导联关系建模 → 多尺度时序融合 → 深层判别"** 的层次化设计思想。

### 4.1 架构总览

```
Input (B, 12, 2000)
│
├─► MultiScaleStem (并行 3/5/7 卷积) ──► (B, 64, 2000)
│
├─► ResConvBlock + SE × 2 ──► (B, 256, 500)
│       │
│       └─► CrossLeadAttention (导联间 Self-Attention)
│
├─► 早期时序分支 (500 长度)
│       ├─► 1×1 Conv 降维 ──► BiGRU(128) + Attention ──► f_e (B, 256)
│
├─► ResConvBlock × 2 ──► (B, 256, 125)
│
└─► 晚期时序分支 (125 长度)
        ├─► BiGRU(512, 2-layer) + Attention ──► f_l (B, 512)
│
Concat[f_e, f_l] ──► 深层分类头 (256+1024 → 512 → 256 → 128 → 1)
```

### 4.2 核心创新模块

#### ① 多尺度初始层（MultiScaleStem）
并行使用 `3×1`、`5×1`、`7×1` 三种尺度的 1D 卷积，在输入层即融合不同感受野的信息，有效捕获 QRS 波的尖锐形态（小核）与 ST-T 段的平缓变化（大核）。

#### ② 残差卷积 + SE 通道注意力（ResConvBlock + SEBlock）
每个残差块内嵌 SE（Squeeze-and-Excitation）注意力，自适应地重新校准各通道特征响应，增强对缺血相关波形（如 ST 段压低、T 波倒置）的敏感性。

#### ③ 导联间关系建模（CrossLeadAttentionModule）
将 256 维通道特征映射到 12 个导联的语义空间（每导联 16 维），通过 `MultiheadAttention` 学习导联间的全局依赖关系（如对前壁心梗的协同响应、肢体导联与胸导联的电传导耦合），再映射回通道空间，实现可解释的导联间信息交互。

#### ④ 多尺度时序金字塔（Multi-Scale Temporal Pyramid）
- **早期分支**（500 时间步）：聚焦局部波形形态（QRS 宽度、ST 段偏移、T 波形态），通过轻量 BiGRU 提取精细时空特征；
- **晚期分支**（125 时间步）：聚焦长程节律模式（心律不齐、传导阻滞），通过深层 BiGRU 捕获全局动态；
- 双分支特征拼接后送入深层分类头，实现局部细节与全局节律的协同判别。

#### ⑤ 深层分类头
采用 `LayerNorm + Linear + ReLU + Dropout` 的多层结构，逐步降维并增强非线性判别能力，最终输出单节点 logits 用于二分类。

---

## 5. 模型权重
文件夹`/ecg_crnn_output`内的五折交叉的模型权重可前往：
```
通过网盘分享的文件：ECG_ischemic_classification_model_weights
链接: https://pan.baidu.com/s/1s7o3ogZ2f0T1kLLKVsiEcQ?pwd=jf6c 提取码: jf6c 
--来自百度网盘超级会员v9的分享
```
进行获取。

---

## 6. 许可证

本项目采用 [MIT License](LICENSE) 开源。
