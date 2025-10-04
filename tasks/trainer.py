# tasks/trainer.py
# -*- coding: utf-8 -*-
"""
학습 엔트리포인트 (torchrun 지원 / 1GPU & DDP 멀티GPU)
- YAML 설정 파일을 인자로 받음
- 모델/옵티마이저/스케줄러/로더 구성
- _train_one_epoch / _validate_one_epoch 메서드 분리
- AdamW 기본, 옵션으로 Adam/SGD/RAdam/Adagrad 등 선택
- 스케줄러: LambdaLR, StepLR, MultiStepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, OneCycleLR, ExponentialLR, ReduceLROnPlateau 지원

실행 예시:
torchrun --standalone --nnodes=1 --nproc_per_node=4 tasks/trainer.py --config ./configs/example.yaml
또는 (1GPU)
python tasks/trainer.py --config ./configs/example.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import math
import logging
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast

# ------------------------------
# 안전한 dataloader 모듈 import (파일명에 '.' 이 있는 요구사항 대응)
# ------------------------------
def _import_dataloader_module() -> Any:
    """
    dataloaders/dataloader_v1.0.py 를 안전하게 importlib로 로드.
    반환: 모듈 객체 (DataloaderHelper, LoaderConfig, ... 사용 가능)
    """
    root = Path(__file__).resolve().parents[1]
    dl_path = root / "dataloaders" / "dataloader_v1.0.py"
    if not dl_path.exists():
        raise FileNotFoundError(f"Not found: {dl_path}")
    spec = importlib.util.spec_from_file_location("dataloader_v1_0", str(dl_path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore
    return module


# ------------------------------
# 구성/유틸/OOP 헬퍼
# ------------------------------
@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "auto"  # auto | cpu | cuda
    epochs: int = 20
    log_interval: int = 50
    amp: bool = True
    amp_dtype: str = "bf16"  # fp16 | bf16
    grad_accum_steps: int = 1
    clip_grad_norm: Optional[float] = 1.0
    cudnn_benchmark: bool = True
    save_every: int = 1


@dataclass
class OptimConfig:
    name: str = "AdamW"  # AdamW | Adam | SGD | RAdam | Adagrad
    lr: float = 5e-4
    weight_decay: float = 1e-2
    betas: Tuple[float, float] = (0.9, 0.999)
    momentum: float = 0.9  # for SGD
    nesterov: bool = False


@dataclass
class SchedConfig:
    name: Optional[str] = "cosine"  # None | lambda | step | multistep | cosine | cosine_restart | onecycle | exponential | reduce_on_plateau
    # 공통/옵션 파라미터
    step_size: int = 5
    gamma: float = 0.5
    milestones: Optional[list] = None
    min_lr: float = 1e-6
    T_max: int = 50
    T_0: int = 10
    eta_min: float = 1e-6
    warmup_steps: int = 0
    cycle_mul: float = 2.0
    # OneCycleLR
    max_lr: Optional[float] = None
    pct_start: float = 0.3
    # ReduceLROnPlateau
    mode: str = "min"
    factor: float = 0.5
    patience: int = 5
    threshold: float = 1e-4


@dataclass
class LossConfig:
    name: str = "mse"  # mse | smoothl1 | l1
    smoothl1_beta: float = 1.0


@dataclass
class DDPConfig:
    backend: str = "nccl"  # nccl | gloo
    find_unused_parameters: bool = False


def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def setup_logger(log_dir: Path, rank: int = 0) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("trainer")
    logger.setLevel(logging.DEBUG if rank == 0 else logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s][%(levelname)s][%(name)s] %(message)s")

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO if rank == 0 else logging.ERROR)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if rank == 0:
        fh = logging.FileHandler(log_dir / "train.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def set_seed(seed: int, rank: int = 0):
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    import random, numpy as np
    random.seed(seed + rank)
    np.random.seed(seed + rank)


class DDPManager:
    """
    torchrun 환경 변수(RANK, LOCAL_RANK, WORLD_SIZE) 기반 초기화
    """
    def __init__(self, ddp_cfg: DDPConfig):
        self.ddp_cfg = ddp_cfg
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.is_distributed = self.world_size > 1
        self.device = torch.device("cpu")

    def setup(self):
        if self.is_distributed:
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend=self.ddp_cfg.backend, init_method="env://")
            self.device = torch.device("cuda", self.local_rank)
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")

    def barrier(self):
        if self.is_distributed and dist.is_initialized():
            dist.barrier()

    def cleanup(self):
        if self.is_distributed and dist.is_initialized():
            dist.destroy_process_group()


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.cnt = 0
        self.avg = 0.0

    def update(self, val: float, n: int = 1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / max(self.cnt, 1)


def build_optimizer(model: torch.nn.Module, cfg: OptimConfig) -> torch.optim.Optimizer:
    name = cfg.name.lower()
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)
    elif name == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)
    elif name == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, nesterov=cfg.nesterov, weight_decay=cfg.weight_decay)
    elif name == "radam":
        return torch.optim.RAdam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif name == "adagrad":
        return torch.optim.Adagrad(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.name}")


@dataclass
class SchedulerObject:
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]
    step_on_batch: bool
    needs_val_metric: bool


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: SchedConfig,
    steps_per_epoch: int,
    total_epochs: int
) -> SchedulerObject:
    if cfg.name in [None, "none", "null"]:
        return SchedulerObject(None, step_on_batch=False, needs_val_metric=False)

    name = cfg.name.lower()
    step_on_batch = False
    needs_val_metric = False

    if name == "lambda":
        def lr_lambda(step):
            # 간단한 warmup + cosine decay 예시
            warmup = max(cfg.warmup_steps, 1)
            if step < warmup:
                return float(step + 1) / float(warmup)
            progress = (step - warmup) / max(1, (steps_per_epoch * total_epochs - warmup))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        step_on_batch = True

    elif name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    elif name == "multistep":
        milestones = cfg.milestones or [30, 60, 90]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=cfg.gamma)
    elif name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.T_max, eta_min=cfg.eta_min)
    elif name == "cosine_restart":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg.T_0, T_mult=cfg.cycle_mul, eta_min=cfg.eta_min)
        step_on_batch = False
    elif name == "onecycle":
        max_lr = cfg.max_lr or max([g["lr"] for g in optimizer.param_groups])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            steps_per_epoch=steps_per_epoch,
            epochs=total_epochs,
            pct_start=cfg.pct_start,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=max_lr / max(cfg.min_lr, 1e-8)
        )
        step_on_batch = True
    elif name == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.gamma)
    elif name == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=cfg.mode,
            factor=cfg.factor,
            patience=cfg.patience,
            threshold=cfg.threshold,
            min_lr=cfg.min_lr
        )
        needs_val_metric = True
    else:
        raise ValueError(f"Unknown scheduler: {cfg.name}")

    return SchedulerObject(scheduler, step_on_batch=step_on_batch, needs_val_metric=needs_val_metric)


def build_loss(cfg: LossConfig):
    name = cfg.name.lower()
    if name == "mse":
        return torch.nn.MSELoss()
    elif name == "smoothl1":
        return torch.nn.SmoothL1Loss(beta=cfg.smoothl1_beta)
    elif name == "l1":
        return torch.nn.L1Loss()
    else:
        raise ValueError(f"Unknown loss: {cfg.name}")


# ------------------------------
# Trainer
# ------------------------------
class Trainer:
    def __init__(self, args: argparse.Namespace):
        raw_cfg = load_yaml_config(args.config)

        # 학습/모델/데이터 관련 서브-설정 추출
        self.work_dir = Path(raw_cfg.get("work_dir", "./work_dir"))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.work_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # dataloader 모듈 import
        dlmod = _import_dataloader_module()
        self.LoaderConfig = dlmod.LoaderConfig
        self.DataloaderHelper = dlmod.DataloaderHelper

        # 구성 dataclass(or dict) 생성
        self.train_cfg = TrainConfig(**raw_cfg.get("train", {}))
        self.optim_cfg = OptimConfig(**raw_cfg.get("optimizer", {}))
        self.sched_cfg = SchedConfig(**raw_cfg.get("scheduler", {}))
        self.loss_cfg = LossConfig(**raw_cfg.get("loss", {}))
        self.ddp_cfg = DDPConfig(**raw_cfg.get("ddp", {}))
        self.model_cfg_dict: Dict[str, Any] = raw_cfg.get("model", {})
        self.data_cfg: Dict[str, Any] = raw_cfg.get("data", {})

        # DDP 설정
        self.ddp = DDPManager(self.ddp_cfg)
        self.ddp.setup()

        # Logger/seeds
        self.logger = setup_logger(self.work_dir, rank=self.ddp.rank)
        set_seed(self.train_cfg.seed, rank=self.ddp.rank)
        torch.backends.cudnn.benchmark = self.train_cfg.cudnn_benchmark

        # 모델/로더/학습자 준비
        self.device = self.ddp.device
        self.model, self.criterion, self.optimizer = self._build_model_optimizer()
        self.scaler = GradScaler(enabled=self.train_cfg.amp)
        self.loaders = self._build_dataloaders()
        self.scheduler_obj = self._build_scheduler()

        # resume
        self.start_epoch = 1
        self.best_val = float("inf")
        resume_path = raw_cfg.get("resume_from", None)
        if resume_path:
            self._load_checkpoint(resume_path)

    # --------------------------
    # Builds
    # --------------------------
    def _build_model_optimizer(self):
        # 모델 import
        from models.transformer_prediction_model import TransformerPredictionModel, TransformerPredictionConfig

        model_cfg = TransformerPredictionConfig(**self.model_cfg_dict)
        model = TransformerPredictionModel(model_cfg).to(self.device)

        # DDP wrapping
        if self.ddp.is_distributed:
            model = DDP(
                model,
                device_ids=[self.ddp.local_rank],
                output_device=self.ddp.local_rank,
                find_unused_parameters=self.ddp_cfg.find_unused_parameters,
            )

        criterion = build_loss(self.loss_cfg)
        optimizer = build_optimizer(model, self.optim_cfg)
        return model, criterion, optimizer

    def _build_dataloaders(self) -> Dict[str, Optional[DataLoader]]:
        # LoaderConfig dataclass로 변환
        def to_loader_cfg(d: Dict[str, Any]) -> Any:
            return self.LoaderConfig(**d)

        train_cfg = to_loader_cfg(self.data_cfg.get("train")) if self.data_cfg.get("train") else None
        val_cfg = to_loader_cfg(self.data_cfg.get("val")) if self.data_cfg.get("val") else None
        test_cfg = to_loader_cfg(self.data_cfg.get("test")) if self.data_cfg.get("test") else None

        helper = self.DataloaderHelper(is_distributed=self.ddp.is_distributed, seed=self.train_cfg.seed)
        loaders = helper.build(train_cfg=train_cfg, val_cfg=val_cfg, test_cfg=test_cfg)
        return loaders

    def _build_scheduler(self) -> SchedulerObject:
        steps_per_epoch = len(self.loaders["train"]) if self.loaders["train"] is not None else 1
        sched_obj = build_scheduler(
            optimizer=self.optimizer,
            cfg=self.sched_cfg,
            steps_per_epoch=steps_per_epoch,
            total_epochs=self.train_cfg.epochs
        )
        return sched_obj

    # --------------------------
    # Checkpoint IO
    # --------------------------
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        if self.ddp.rank != 0:
            return
        to_save = self.model.module if isinstance(self.model, DDP) else self.model
        state = {
            "epoch": epoch,
            "model": to_save.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "best_val": self.best_val,
            "sched": self.scheduler_obj.scheduler.state_dict() if self.scheduler_obj.scheduler is not None else None
        }
        path = self.ckpt_dir / f"epoch_{epoch:04d}.pth"
        torch.save(state, path)
        if is_best:
            torch.save(state, self.ckpt_dir / "best.pth")
        self.logger.info(f"Checkpoint saved: {path}{' (best)' if is_best else ''}")

    def _load_checkpoint(self, path: str):
        mp = torch.load(path, map_location="cpu")
        if isinstance(self.model, DDP):
            self.model.module.load_state_dict(mp["model"])
        else:
            self.model.load_state_dict(mp["model"])
        self.optimizer.load_state_dict(mp["optimizer"])
        if "scaler" in mp and mp["scaler"] is not None:
            self.scaler.load_state_dict(mp["scaler"])
        if self.scheduler_obj.scheduler is not None and "sched" in mp and mp["sched"] is not None:
            self.scheduler_obj.scheduler.load_state_dict(mp["sched"])
        self.start_epoch = mp.get("epoch", 1) + 1
        self.best_val = mp.get("best_val", float("inf"))
        self.logger.info(f"Resumed from {path} (epoch {self.start_epoch-1}, best {self.best_val:.6f})")

    # --------------------------
    # Epoch loops
    # --------------------------
    def _train_one_epoch(self, epoch: int) -> float:
        """한 에포크 학습 루프 (AMP/Grad Accum/DDP 지원)."""
        self.model.train()
        loader = self.loaders["train"]
        assert loader is not None, "Train loader is None"
        if self.ddp.is_distributed and isinstance(loader.sampler, torch.utils.data.distributed.DistributedSampler):
            loader.sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        step_global = 0
        accum = max(1, self.train_cfg.grad_accum_steps)

        dtype = torch.bfloat16 if (self.train_cfg.amp and self.train_cfg.amp_dtype.lower() == "bf16") else torch.float16
        for step, batch in enumerate(loader, start=1):
            x = batch["x"].to(self.device, non_blocking=True)
            y = batch["y"].to(self.device, non_blocking=True)
            lengths = batch["lengths"].to(self.device, non_blocking=True)

            with autocast(enabled=self.train_cfg.amp, dtype=dtype):
                preds = self.model(x, lengths=lengths)
                loss = self.criterion(preds, y)
                loss = loss / accum

            self.scaler.scale(loss).backward()

            if step % accum == 0:
                if self.train_cfg.clip_grad_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        (self.model.module if isinstance(self.model, DDP) else self.model).parameters(),
                        self.train_cfg.clip_grad_norm
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                # 스케줄러 per-step
                if self.scheduler_obj.scheduler is not None and self.scheduler_obj.step_on_batch:
                    self.scheduler_obj.scheduler.step()

                step_global += 1

            loss_meter.update(loss.item() * accum, n=x.size(0))

            if self.ddp.rank == 0 and (step % self.train_cfg.log_interval == 0):
                lr = self.optimizer.param_groups[0]["lr"]
                self.logger.info(f"[Epoch {epoch}][Step {step}/{len(loader)}] "
                                 f"loss={loss_meter.avg:.6f} lr={lr:.6e}")

        return loss_meter.avg

    @torch.no_grad()
    def _validate_one_epoch(self, epoch: int) -> float:
        """검증 루프 (DDP에서 rank0만 로그 출력)."""
        loader = self.loaders["val"]
        if loader is None:
            return float("nan")

        self.model.eval()
        loss_meter = AverageMeter()

        dtype = torch.bfloat16 if (self.train_cfg.amp and self.train_cfg.amp_dtype.lower() == "bf16") else torch.float16
        for step, batch in enumerate(loader, start=1):
            x = batch["x"].to(self.device, non_blocking=True)
            y = batch["y"].to(self.device, non_blocking=True)
            lengths = batch["lengths"].to(self.device, non_blocking=True)

            with autocast(enabled=self.train_cfg.amp, dtype=dtype):
                preds = self.model(x, lengths=lengths)
                loss = self.criterion(preds, y)

            loss_meter.update(loss.item(), n=x.size(0))

        # 스케줄러 per-epoch
        if self.scheduler_obj.scheduler is not None and not self.scheduler_obj.step_on_batch:
            if self.scheduler_obj.needs_val_metric:
                self.scheduler_obj.scheduler.step(loss_meter.avg)
            else:
                self.scheduler_obj.scheduler.step()

        if self.ddp.rank == 0:
            self.logger.info(f"[Validate {epoch}] val_loss={loss_meter.avg:.6f}")
        return loss_meter.avg

    # --------------------------
    # Fit
    # --------------------------
    def fit(self):
        best = self.best_val
        for epoch in range(self.start_epoch, self.train_cfg.epochs + 1):
            t0 = time.time()
            train_loss = self._train_one_epoch(epoch)
            val_loss = self._validate_one_epoch(epoch)
            dt = time.time() - t0

            if self.ddp.rank == 0:
                self.logger.info(f"[Epoch {epoch} done] train_loss={train_loss:.6f} val_loss={val_loss:.6f} ({dt:.1f}s)")

            # best 저장
            improved = (val_loss < best) if not math.isnan(val_loss) else False
            if improved:
                best = val_loss
            if (epoch % self.train_cfg.save_every == 0) or improved:
                self._save_checkpoint(epoch, is_best=improved)

        if self.ddp.rank == 0:
            self.logger.info(f"Training finished. Best val_loss={best:.6f}")

    # --------------------------
    # __del__
    # --------------------------
    def __del__(self):
        try:
            self.ddp.cleanup()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="YAML config path")
    return p.parse_args()


def main():
    args = parse_args()
    trainer = Trainer(args)
    trainer.fit()


if __name__ == "__main__":
    main()