# tasks/inference.py
# -*- coding: utf-8 -*-
"""
Inference 엔트리포인트
- YAML config + checkpoint 경로 입력
- dataloader_v1.0.py를 importlib로 안전 로드
- AMP 지원
- 예측 결과를 CSV로 저장

예시:
python tasks/inference.py --config ./configs/example.yaml --checkpoint ./work_dir/checkpoints/best.pth --output ./work_dir/preds.csv
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
import torch
from torch.cuda.amp import autocast


def _import_dataloader_module() -> Any:
    root = Path(__file__).resolve().parents[1]
    dl_path = root / "dataloaders" / "dataloader_v1.0.py"
    spec = importlib.util.spec_from_file_location("dataloader_v1_0", str(dl_path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore
    return module


def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


class Inferencer:
    def __init__(self, cfg: Dict[str, Any], checkpoint: str, output_path: str):
        self.cfg = cfg
        self.output_path = Path(output_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # dataloader module
        dlmod = _import_dataloader_module()
        self.LoaderConfig = dlmod.LoaderConfig
        self.DataloaderHelper = dlmod.DataloaderHelper

        # build model
        from models.transformer_prediction_model import TransformerPredictionModel, TransformerPredictionConfig

        model_cfg = TransformerPredictionConfig(**cfg["model"])
        self.model = TransformerPredictionModel(model_cfg).to(self.device)

        state = torch.load(checkpoint, map_location="cpu")
        self.model.load_state_dict(state["model"])
        self.model.eval()

        # build loader (test split 필수)
        data_cfg = cfg["data"]["test"]
        test_cfg = self.LoaderConfig(**data_cfg)
        helper = self.DataloaderHelper(is_distributed=False, seed=cfg.get("train", {}).get("seed", 42))
        loaders = helper.build(train_cfg=None, val_cfg=None, test_cfg=test_cfg)
        self.loader = loaders["test"]
        assert self.loader is not None, "Test loader is None"

        self.amp = cfg.get("train", {}).get("amp", True)
        self.amp_dtype = cfg.get("train", {}).get("amp_dtype", "bf16")

    @torch.no_grad()
    def predict(self):
        rows = []
        dtype = torch.bfloat16 if (self.amp and self.amp_dtype.lower() == "bf16") else torch.float16

        for batch in self.loader:
            x = batch["x"].to(self.device, non_blocking=True)
            lengths = batch["lengths"].to(self.device, non_blocking=True)
            ids = batch["ids"]

            with autocast(enabled=self.amp, dtype=dtype):
                preds = self.model(x, lengths=lengths)  # (B, K)
            preds = preds.detach().cpu().numpy()

            for _id, p in zip(ids, preds):
                # 출력은 첫 컬럼이 id, 이후 각 출력 차원
                row = {"id": _id}
                for i, v in enumerate(p.tolist()):
                    row[f"pred_{i}"] = v
                rows.append(row)

        # CSV 저장
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = rows[0].keys() if rows else ["id", "pred_0"]
        with self.output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

        print(f"Saved predictions to: {self.output_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml_config(args.config)
    Inferencer(cfg, args.checkpoint, args.output).predict()


if __name__ == "__main__":
    main()