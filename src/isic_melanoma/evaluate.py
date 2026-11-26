"""Dự đoán và tính metric.

Đặt file này tại: src/<package_name>/evaluate.py

Metric không phụ thuộc ngưỡng: ROC-AUC, PR-AUC, sensitivity tại specificity cố định.
Metric phụ thuộc ngưỡng (chỉ tính khi truyền `threshold`): sensitivity, specificity,
precision, F1, balanced accuracy, confusion matrix. Ngưỡng nên chọn trên VALIDATION
(youden_threshold) rồi áp dụng nguyên cho test.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve


@torch.no_grad()
def predict(model: torch.nn.Module, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Trả về (nhãn thật, xác suất ác tính) theo đúng thứ tự của loader."""
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True)).squeeze(1)
        ps.append(torch.sigmoid(logits).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Ngưỡng tối đa hóa (sensitivity + specificity - 1). Rơi về 0.5 nếu không xác định được."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    thr = float(thresholds[np.argmax(tpr - fpr)])
    return thr if np.isfinite(thr) else 0.5


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else float("nan")


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float | None = None,
    target_specificity: float = 0.90,
) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    sens_key = f"sens_at_spec{int(round(target_specificity * 100))}"
    out: dict[str, float] = {}

    if len(np.unique(y_true)) < 2:  # chỉ có một lớp: các metric xếp hạng không xác định
        out.update({"auc": float("nan"), "pr_auc": float("nan"), sens_key: float("nan")})
    else:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ok = fpr <= (1.0 - target_specificity)
        out["auc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
        out[sens_key] = float(tpr[ok].max())

    if threshold is not None:
        pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        sens = _safe_div(tp, tp + fn)
        spec = _safe_div(tn, tn + fp)
        prec = _safe_div(tp, tp + fp)
        out.update(
            {
                "threshold": float(threshold),
                "sensitivity": sens,
                "specificity": spec,
                "precision": prec,
                "f1": _safe_div(2 * prec * sens, prec + sens) if prec == prec and sens == sens else float("nan"),
                "balanced_acc": (sens + spec) / 2,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
    return out
