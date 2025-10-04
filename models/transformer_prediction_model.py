# models/transformer_prediction_model.py
# -*- coding: utf-8 -*-
"""
Transformer Encoder 기반 회귀 모델.
- 연속형 시퀀스 입력 x: (B, T, F)
- 마스킹/패딩 처리
- 포지셔널 인코딩(Sinusoidal)
- Pooling: 'mean' | 'cls' | 'last'
- Regression head를 통해 스칼라(또는 다중) 회귀 출력

Author: Your Name
"""

from dataclasses import dataclass
from typing import Optional, Literal, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerPredictionConfig:
    input_dim: int                  # 입력 피처 차원(F)
    d_model: int = 256              # 임베딩 차원
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    activation: Literal["relu", "gelu"] = "gelu"
    norm_first: bool = True
    use_cls_token: bool = True
    pooling: Literal["mean", "cls", "last"] = "mean"
    max_len: int = 4096
    output_dim: int = 1             # 회귀 출력 차원 (기본 1)
    input_norm: bool = False        # 입력 정규화 레이어 사용할지 여부


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)  # (T, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd
        pe = pe.unsqueeze(0)  # (1, T, d_model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, d_model)
        """
        if x.size(1) > self.pe.size(1):
            raise ValueError(
                f"[PositionalEncoding] sequence length {x.size(1)} "
                f"exceeds max_len {self.pe.size(1)}"
            )
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class RegressionHead(nn.Module):
    def __init__(self, d_model: int, output_dim: int = 1, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerPredictionModel(nn.Module):
    """
    Transformer Encoder 기반 회귀 모델 (batch_first=True).
    """
    def __init__(self, cfg: TransformerPredictionConfig):
        super().__init__()
        self.cfg = cfg

        self.input_norm = nn.LayerNorm(cfg.input_dim) if cfg.input_norm else nn.Identity()
        self.input_proj = nn.Linear(cfg.input_dim, cfg.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation=cfg.activation,
            batch_first=True,
            norm_first=cfg.norm_first
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.num_layers, norm=nn.LayerNorm(cfg.d_model)
        )
        self.pos_encoder = PositionalEncoding(cfg.d_model, cfg.max_len, dropout=0.0)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model)) if cfg.use_cls_token else None
        nn.init.trunc_normal_(self.cls_token, std=0.02) if self.cls_token is not None else None

        self.head = RegressionHead(cfg.d_model, cfg.output_dim, dropout=cfg.dropout)

    @staticmethod
    def _make_src_key_padding_mask(lengths: Optional[torch.Tensor], max_len: int) -> Optional[torch.Tensor]:
        """
        Returns mask with shape (B, T) where True indicates padding positions to mask.
        """
        if lengths is None:
            return None
        device = lengths.device
        rng = torch.arange(max_len, device=device).unsqueeze(0)  # (1, T)
        mask = rng >= lengths.unsqueeze(1)                       # (B, T)
        return mask

    def _pool(self, x: torch.Tensor, mask: Optional[torch.Tensor], lengths: Optional[torch.Tensor]) -> torch.Tensor:
        """
        x: (B, T, C), mask: (B, T) True=pad, lengths: (B,)
        """
        if self.cfg.pooling == "cls" and self.cls_token is not None:
            return x[:, 0, :]  # [CLS] 위치

        if self.cfg.pooling == "last":
            if lengths is None:
                raise ValueError("Pooling 'last' requires 'lengths' tensor.")
            # 마지막 유효 토큰 인덱스
            idx = torch.clamp(lengths - 1, min=0)  # (B,)
            return x[torch.arange(x.size(0), device=x.device), idx, :]

        # default: mean pooling (유효 토큰만 평균)
        if mask is None:
            return x.mean(dim=1)
        valid = (~mask).float()  # (B, T)
        denom = torch.clamp(valid.sum(dim=1, keepdim=True), min=1.0)
        return (x * valid.unsqueeze(-1)).sum(dim=1) / denom

    def forward(
        self,
        inputs: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        inputs: (B, T, F), lengths: (B,)
        returns: predictions (B, output_dim)
        """
        B, T, _ = inputs.shape
        x = self.input_norm(inputs)
        x = self.input_proj(x)
        # [CLS] 토큰 앞에 붙이기 (옵션)
        if self.cls_token is not None:
            cls = self.cls_token.expand(B, 1, -1)
            x = torch.cat([cls, x], dim=1)
            T = T + 1
            if lengths is not None:
                lengths = lengths + 1

        # Positional encoding
        x = self.pos_encoder(x)

        # Padding mask
        src_key_padding_mask = self._make_src_key_padding_mask(lengths, T)  # (B, T) True=mask

        # Encoder
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Pooling
        pooled = self._pool(x, src_key_padding_mask, lengths)

        # Head
        out = self.head(pooled)
        return out