"""
eegnet.py — EEGNet: Compact CNN for EEG-Based Brain Signal Classification

Reference: Lawhern et al. (2018) — EEGNet: a compact CNN for EEG-based BCIs

Architecture:
    Input  : (B, 1, T)  — single-channel EEG window
    Block 1: Temporal Conv1d → Depthwise Conv1d (spatial) → BN → ELU → Pool → Dropout
    Block 2: Separable Conv1d (Depthwise + Pointwise) → BN → ELU → Pool → Dropout
    Head   : AdaptiveAvgPool → Linear → Sigmoid

Adaptive pooling makes the model input-length agnostic.
"""

import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,   # EEG channels (1 for single-channel)
        F1: int = 8,             # Temporal filters
        D: int = 2,              # Depth multiplier
        F2: int = 16,            # Separable conv filters
        kernel_length: int = 64, # Temporal filter length (≈ 0.25 s at 256 Hz)
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        F_D = F1 * D  # filters after depthwise

        # ── Block 1: Temporal + Depthwise ─────────────────────────────────
        self.temporal = nn.Sequential(
            nn.Conv1d(in_channels, F1, kernel_size=kernel_length,
                      padding=kernel_length // 2, bias=False),
            nn.BatchNorm1d(F1),
        )
        self.depthwise = nn.Sequential(
            # Depthwise captures channel-spatial interactions
            nn.Conv1d(F1, F_D, kernel_size=1, groups=F1, bias=False),
            nn.BatchNorm1d(F_D),
            nn.ELU(inplace=True),
            nn.AvgPool1d(kernel_size=4),
            nn.Dropout(dropout),
        )

        # ── Block 2: Separable (Depthwise + Pointwise) ────────────────────
        self.separable = nn.Sequential(
            nn.Conv1d(F_D, F_D, kernel_size=16, padding=8,
                      groups=F_D, bias=False),           # depthwise
            nn.Conv1d(F_D, F2, kernel_size=1, bias=False),  # pointwise
            nn.BatchNorm1d(F2),
            nn.ELU(inplace=True),
            nn.AvgPool1d(kernel_size=8),
            nn.Dropout(dropout),
        )

        # ── Classifier ────────────────────────────────────────────────────
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc  = nn.Linear(F2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T) or (B, 1, T)
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 2:
            x = x.unsqueeze(1)              # (B, 1, T)

        x = self.temporal(x)               # (B, F1, T)
        x = self.depthwise(x)              # (B, F1*D, T//4)
        x = self.separable(x)             # (B, F2,   T//32)
        x = self.gap(x).squeeze(-1)        # (B, F2)
        x = self.fc(x)                     # (B, 1)
        return torch.sigmoid(x).squeeze(-1)  # (B,)
