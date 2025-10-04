# dataloaders/dataloader_v1.0.py
# -*- coding: utf-8 -*-
"""
시퀀스 회귀용 Dataset + Collator + DataloaderHelper
- CSV/JSONL 간단 지원
- sequence_col: 각 행의 시퀀스(리스트[list] of list[float])를 JSON 문자열 형태로 보관하거나
               파이썬 literal 형식 문자열로 보관 가능. (예: "[[0.1,0.2],[0.3,0.4]]")
- target_col: 회귀 타깃(단일 or 다중), 없으면 Inference용으로 처리
- id_col: 식별자(optional)
- collate 시 pad_sequence + mask/lengths 생성
"""

from __future__ import annotations

import ast
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, DistributedSampler


def _safe_parse_sequence(text: Union[str, List[List[float]]]) -> List[List[float]]:
    """
    문자열 또는 이미 파싱된 리스트를 안전하게 2D float 리스트로 변환.
    """
    if isinstance(text, list):
        return text
    s = text.strip()
    # 우선 JSON -> 실패하면 literal_eval
    try:
        val = json.loads(s)
    except json.JSONDecodeError:
        val = ast.literal_eval(s)
    if not isinstance(val, list) or (len(val) > 0 and not isinstance(val[0], list)):
        raise ValueError(f"Sequence format must be list of list, got: {type(val)} {val[:50] if isinstance(val, list) else val}")
    return val


def _to_tensor2d(arr: List[List[float]]) -> torch.Tensor:
    return torch.tensor(arr, dtype=torch.float32)


class SequenceRegressionDataset(Dataset):
    """
    CSV 혹은 JSONL 파일에서 시퀀스/타깃을 읽어오는 Dataset.
    """
    def __init__(
        self,
        path: Union[str, Path],
        filetype: str,
        sequence_col: str,
        target_col: Optional[Union[str, List[str]]] = None,
        id_col: Optional[str] = None,
        strict: bool = True,
    ):
        super().__init__()
        self.path = Path(path)
        self.filetype = filetype.lower()
        self.sequence_col = sequence_col
        self.target_col = target_col
        self.id_col = id_col
        self.strict = strict

        if self.filetype not in {"csv", "jsonl"}:
            raise ValueError(f"Unsupported filetype: {self.filetype}")

        self.rows: List[Dict[str, Any]] = self._load_rows()

    def _load_rows(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        if self.filetype == "csv":
            with self.path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    rows.append(r)
        else:  # jsonl
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))
        if len(rows) == 0 and self.strict:
            raise ValueError(f"No data loaded from {self.path}")
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        seq = _safe_parse_sequence(row[self.sequence_col])
        x = _to_tensor2d(seq)  # (T, F)

        y = None
        if self.target_col is not None:
            if isinstance(self.target_col, list):
                y = torch.tensor([float(row[c]) for c in self.target_col], dtype=torch.float32)  # (K,)
            else:
                y = torch.tensor(float(row[self.target_col]), dtype=torch.float32).unsqueeze(0)  # (1,)

        sample = {
            "x": x,
            "y": y,
            "id": row[self.id_col] if self.id_col and self.id_col in row else idx,
            "length": x.size(0),
        }
        return sample


class PadCollator:
    """
    variable-length batch -> padded batch (+ lengths, mask)
    """
    def __init__(self, pad_value: float = 0.0, return_mask: bool = True):
        self.pad_value = pad_value
        self.return_mask = return_mask

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        xs = [b["x"] for b in batch]  # (Ti, F)
        ys = [b["y"] for b in batch]  # None or (K,)
        ids = [b["id"] for b in batch]
        lens = torch.tensor([b["length"] for b in batch], dtype=torch.long)

        padded_x = pad_sequence(xs, batch_first=True, padding_value=self.pad_value)  # (B, T, F)
        if ys[0] is None:
            y = None
        else:
            y = torch.stack(ys, dim=0)  # (B, K)

        max_len = padded_x.size(1)
        mask = (torch.arange(max_len).unsqueeze(0) >= lens.unsqueeze(1))  # (B, T), True=pad

        out = {
            "x": padded_x,
            "y": y,
            "lengths": lens,
            "mask": mask,
            "ids": ids,
        }
        return out


@dataclass
class LoaderConfig:
    path: str
    filetype: str
    sequence_col: str
    target_col: Optional[Union[str, List[str]]] = None
    id_col: Optional[str] = None
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True
    drop_last: bool = False
    pin_memory: bool = True
    pad_value: float = 0.0
    persistent_workers: bool = True


class DataloaderHelper:
    """
    설정(LoaderConfig dict-like)으로 Train/Val/Test DataLoader 구성.
    DDP 시 DistributedSampler 자동 적용.
    """
    def __init__(self, is_distributed: bool = False, seed: int = 42):
        self.is_distributed = is_distributed
        self.seed = seed

    @staticmethod
    def _build_dataset(cfg: LoaderConfig) -> SequenceRegressionDataset:
        return SequenceRegressionDataset(
            path=cfg.path,
            filetype=cfg.filetype,
            sequence_col=cfg.sequence_col,
            target_col=cfg.target_col,
            id_col=cfg.id_col,
            strict=True
        )

    def _build_loader(
        self,
        cfg: LoaderConfig,
        dataset: SequenceRegressionDataset,
        is_train: bool
    ) -> DataLoader:
        sampler = None
        if self.is_distributed:
            sampler = DistributedSampler(
                dataset,
                shuffle=cfg.shuffle if is_train else False,
                drop_last=cfg.drop_last if is_train else False,
            )

        collator = PadCollator(pad_value=cfg.pad_value, return_mask=True)

        def _worker_init(worker_id: int):
            # 각 worker마다 seed 다르게
            base_seed = self.seed
            torch.manual_seed(base_seed + worker_id)

        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=(cfg.shuffle and sampler is None and is_train),
            sampler=sampler,
            num_workers=cfg.num_workers,
            drop_last=cfg.drop_last if is_train else False,
            pin_memory=cfg.pin_memory,
            persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
            collate_fn=collator,
            worker_init_fn=_worker_init
        )
        return loader

    def build(
        self,
        train_cfg: Optional[LoaderConfig] = None,
        val_cfg: Optional[LoaderConfig] = None,
        test_cfg: Optional[LoaderConfig] = None
    ) -> Dict[str, Optional[DataLoader]]:
        out: Dict[str, Optional[DataLoader]] = {"train": None, "val": None, "test": None}
        if train_cfg is not None:
            train_ds = self._build_dataset(train_cfg)
            out["train"] = self._build_loader(train_cfg, train_ds, is_train=True)
        if val_cfg is not None:
            val_ds = self._build_dataset(val_cfg)
            out["val"] = self._build_loader(val_cfg, val_ds, is_train=False)
        if test_cfg is not None:
            test_ds = self._build_dataset(test_cfg)
            out["test"] = self._build_loader(test_cfg, test_ds, is_train=False)
        return out