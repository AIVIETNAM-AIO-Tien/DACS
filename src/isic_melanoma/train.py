"""Huấn luyện MỘT lần chạy (một backbone, một seed), chỉ train classifier head.

Đặt file này tại: src/<package_name>/train.py

Nguyên tắc:
- Chọn epoch tốt nhất bằng VALIDATION (cfg.selection_metric), không dùng test.
- Ngưỡng phân loại chọn trên validation (Youden) rồi áp dụng cho test.
- Lưu checkpoint của head và dự đoán val/test để phân tích lỗi, bootstrap CI, Grad-CAM sau này.
"""
from __future__ import annotations

import logging
import math
import time

import pandas as pd
import torch
from torch import nn

from .config import Config
from .data.dataset import make_dataloaders
from .evaluate import compute_metrics, predict, youden_threshold
from .models.factory import (
    build_model,
    count_parameters,
    head_state_dict,
    measure_latency_ms,
    set_head_only_train_mode,
)
from .utils import get_device, get_logger, set_seed


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    set_head_only_train_mode(model)  # BN của backbone giữ ở eval
    total_loss, n = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x).squeeze(1), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / n


def _save_predictions(cfg: Config, backbone: str, seed: int, parts: dict) -> None:
    out_dir = cfg.results_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for split, (loader, y, p) in parts.items():
        frames.append(
            pd.DataFrame(
                {"image_name": loader.dataset.names, "split": split, "y_true": y.astype(int), "y_prob": p}
            )
        )
    pd.concat(frames).to_csv(out_dir / f"{backbone}_seed{seed}.csv", index=False)


def train_one_run(
    cfg: Config,
    backbone: str,
    seed: int,
    augment: bool = True,
    device: torch.device | None = None,
    logger: logging.Logger | None = None,
) -> dict:
    """Huấn luyện và trả về một dòng kết quả (dict phẳng) để ghi vào CSV."""
    device = device or get_device()
    logger = logger or get_logger("train")
    set_seed(seed)

    data = make_dataloaders(cfg, seed, augment)
    model = build_model(backbone).to(device)
    total_params, trainable_params = count_parameters(model)

    pos_weight = torch.tensor([data.pos_weight], device=device) if cfg.use_class_weight else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    metric_key = cfg.selection_metric.removeprefix("val_")  # "val_auc" -> "auc"
    best_score, best_epoch, best_state, bad_epochs = -math.inf, 0, None, 0

    logger.info(f"[{backbone} | seed {seed}] params: {trainable_params:,} huấn luyện / {total_params:,} tổng")
    start = time.perf_counter()
    for epoch in range(1, cfg.max_epochs + 1):
        loss = train_one_epoch(model, data.train, optimizer, criterion, device)
        y_val, p_val = predict(model, data.val, device)
        score = compute_metrics(y_val, p_val)[metric_key]
        logger.info(f"[{backbone} | seed {seed}] epoch {epoch:02d} loss={loss:.4f} val_{metric_key}={score:.4f}")

        if best_state is None or score > best_score:
            best_score, best_epoch, bad_epochs = score, epoch, 0
            best_state = {k: v.clone().cpu() for k, v in head_state_dict(model).items()}
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                logger.info(f"[{backbone} | seed {seed}] early stopping (tốt nhất ở epoch {best_epoch})")
                break
    train_time = time.perf_counter() - start

    model.load_state_dict(best_state, strict=False)  # chỉ nạp lại head
    y_val, p_val = predict(model, data.val, device)
    threshold = youden_threshold(y_val, p_val)  # chọn ngưỡng trên validation
    y_test, p_test = predict(model, data.test, device)
    val_m = compute_metrics(y_val, p_val, threshold)
    test_m = compute_metrics(y_test, p_test, threshold)

    ckpt_dir = cfg.results_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, ckpt_dir / f"{backbone}_seed{seed}.pt")
    _save_predictions(cfg, backbone, seed, {"val": (data.val, y_val, p_val), "test": (data.test, y_test, p_test)})

    row = {
        "backbone": backbone,
        "seed": seed,
        "best_epoch": best_epoch,
        "threshold": threshold,
        "train_time_s": round(train_time, 1),
        "latency_ms": round(measure_latency_ms(model, cfg.img_size, device), 3),
        "params_total": total_params,
        "params_trainable": trainable_params,
    }
    row.update({f"val_{k}": v for k, v in val_m.items() if k != "threshold"})
    row.update({f"test_{k}": v for k, v in test_m.items() if k != "threshold"})
    logger.info(f"[{backbone} | seed {seed}] xong: val_auc={row['val_auc']:.4f} test_auc={row['test_auc']:.4f}")
    return row
