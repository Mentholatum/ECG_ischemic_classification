import torch
import torch.nn as nn
import torch.nn.functional as F


# ========================== 1. SE 通道注意力 ==========================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(1, channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // reduction), channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


# ========================== 2. 残差卷积块 ==========================
class ResConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout1d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.se = SEBlock(out_ch, reduction=max(1, out_ch // 16))
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv(x)
        out = self.se(out)
        out += residual
        return self.relu(out)


# ========================== 3. 多尺度初始层 ==========================
class MultiScaleStem(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        # 3 个尺度并行，输出通道均分
        c1 = out_ch // 3
        c2 = out_ch // 3
        c3 = out_ch - c1 - c2
        self.conv3 = nn.Conv1d(in_ch, c1, 3, padding=1, bias=False)
        self.conv5 = nn.Conv1d(in_ch, c2, 5, padding=2, bias=False)
        self.conv7 = nn.Conv1d(in_ch, c3, 7, padding=3, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout1d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = torch.cat([self.conv3(x), self.conv5(x), self.conv7(x)], dim=1)
        x = self.bn(x)
        x = self.relu(x)
        return self.drop(x)


# ========================== 4. 导联间关系建模 ==========================
class CrossLeadAttentionModule(nn.Module):
    """
    将通道特征映射到 12 导联语义空间，做导联间 Self-Attention，再映射回通道空间。
    可学习地建模 V1-V3 对心梗的协同、肢体导联与胸导联的电传导关系等。
    """
    def __init__(self, in_channels, n_leads=12, lead_dim=16, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_leads = n_leads
        self.lead_dim = lead_dim

        # 映射到导联语义空间: in_channels -> n_leads * lead_dim
        self.to_leads = nn.Conv1d(in_channels, n_leads * lead_dim, 1, bias=False)
        # 导联间自注意力
        self.norm1 = nn.LayerNorm(lead_dim)
        self.attn = nn.MultiheadAttention(lead_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(lead_dim)
        self.ffn = nn.Sequential(
            nn.Linear(lead_dim, lead_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lead_dim * 2, lead_dim),
            nn.Dropout(dropout),
        )
        # 映射回原始通道数
        self.to_channels = nn.Conv1d(n_leads * lead_dim, in_channels, 1, bias=False)
        self.bn = nn.BatchNorm1d(in_channels)

    def forward(self, x):
        B, C, L = x.shape

        # 1. 映射到导联空间: (B, n_leads*lead_dim, L)
        x_lead = self.to_leads(x)

        # 2. 重塑为 (B*L, n_leads, lead_dim) 以并行处理所有时间步
        x_lead = x_lead.permute(0, 2, 1)                       # (B, L, n_leads*lead_dim)
        x_lead = x_lead.reshape(B, L, self.n_leads, self.lead_dim)
        x_lead = x_lead.reshape(B * L, self.n_leads, self.lead_dim)

        # 3. 导联间 Self-Attention
        x_lead = self.norm1(x_lead)
        attn_out, _ = self.attn(x_lead, x_lead, x_lead)
        x_lead = x_lead + attn_out
        x_lead = x_lead + self.ffn(self.norm2(x_lead))

        # 4. 恢复并映射回通道空间
        x_lead = x_lead.reshape(B, L, self.n_leads, self.lead_dim)
        x_lead = x_lead.reshape(B, L, -1).permute(0, 2, 1)     # (B, n_leads*lead_dim, L)
        x = self.to_channels(x_lead)
        x = self.bn(x)
        return x


# ========================== 5. 主模型 ==========================
class ECGNet(nn.Module):
    """
    完整架构：
      - Multi-Scale Stem
      - Residual Blocks + SE
      - Cross-Lead Attention（导联间关系建模）
      - Multi-Scale Temporal Pyramid（500 长度局部 + 125 长度节律）
      - BiGRU (256 hidden) + Attention
      - 深层分类头
    """
    def __init__(self, in_channels=12, n_classes=1, dropout=0.3):
        super().__init__()

        # --- 1. 多尺度初始层 ---
        self.stem = MultiScaleStem(in_channels, 64, dropout=dropout)

        # --- 2. Residual CNN + SE ---
        self.layer1 = ResConvBlock(64, 128, stride=2, dropout=dropout)   # 1000
        self.layer2 = ResConvBlock(128, 256, stride=2, dropout=dropout)  # 500

        # --- 3. 导联间关系建模---
        self.cross_lead = CrossLeadAttentionModule(
            in_channels=256, n_leads=12, lead_dim=16, n_heads=4, dropout=dropout
        )

        self.layer3 = ResConvBlock(256, 256, stride=2, dropout=dropout)  # 250
        self.layer4 = ResConvBlock(256, 256, stride=2, dropout=dropout)  # 125

        # --- 4. 多尺度时序金字塔 ---
        # 早期分支：500 长度，专注 ST-T 改变、QRS 局部形态
        self.early_proj = nn.Sequential(
            nn.Conv1d(256, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.gru_early = nn.GRU(
            input_size=128, hidden_size=128, num_layers=1,
            batch_first=True, bidirectional=True,
        )
        self.att_early = nn.Sequential(
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # 晚期分支：125 长度，专注长程节律（房颤、传导阻滞）
        self.gru_late = nn.GRU(
            input_size=256, hidden_size=512, num_layers=2,
            batch_first=True, bidirectional=True,
            dropout=dropout if dropout > 0 else 0,
        )
        self.att_late = nn.Sequential(
            nn.Linear(1024, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        # --- 5. 深层分类头 ---
        self.classifier = nn.Sequential(
            nn.LayerNorm(256 + 1024),
            nn.Linear(256 + 1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # x: (B, 12, 2000)

        # --- CNN 编码 ---
        x = self.stem(x)          # (B, 64, 2000)
        x = self.layer1(x)        # (B, 128, 1000)
        x = self.layer2(x)        # (B, 256, 500)

        # --- Cross-Lead Attention ---
        x = self.cross_lead(x)    # (B, 256, 500)

        # --- 早期时序分支（500 长度）---
        x_e = self.early_proj(x)  # (B, 128, 500)
        x_e = x_e.permute(0, 2, 1)  # (B, 500, 128)
        x_e, _ = self.gru_early(x_e)  # (B, 500, 256)
        a_e = torch.softmax(self.att_early(x_e), dim=1)  # (B, 500, 1)
        f_e = (x_e * a_e).sum(dim=1)  # (B, 256)

        # --- 继续深层 CNN ---
        x = self.layer3(x)        # (B, 256, 250)
        x = self.layer4(x)        # (B, 256, 125)

        # --- 晚期时序分支（125 长度）---
        x_l = x.permute(0, 2, 1)  # (B, 125, 256)
        x_l, _ = self.gru_late(x_l)  # (B, 125, 512)
        a_l = torch.softmax(self.att_late(x_l), dim=1)  # (B, 125, 1)
        f_l = (x_l * a_l).sum(dim=1)  # (B, 512)

        # --- 融合与分类 ---
        feat = torch.cat([f_e, f_l], dim=-1)  # (B, 768)
        logits = self.classifier(feat)        # (B, 1)
        return logits

    def get_trainable_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable



# ========================== 6. 快速验证 ==========================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGNet(in_channels=12, n_classes=1, dropout=0.3).to(device)

    x = torch.randn(2, 12, 2000).to(device)
    out = model(x)
    total, trainable = model.get_trainable_params()

    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Total params:     {total / 1e6:.3f}M")
    print(f"Trainable params: {trainable / 1e6:.3f}M")