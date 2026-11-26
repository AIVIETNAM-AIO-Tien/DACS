"""Tạo backbone pretrained (đóng băng) + classifier head mới, đầu ra 1 logit.

Đặt file này tại: src/<package_name>/models/factory.py

Lưu ý với chế độ head-only: các lớp BatchNorm của backbone phải ở chế độ eval,
nếu không thống kê BN vẫn bị cập nhật dù trọng số đã đóng băng. Dùng
`set_head_only_train_mode(model)` thay cho `model.train()`.
"""
from __future__ import annotations

import time

import torch
from torch import nn
from torchvision.models import get_model

SUPPORTED_BACKBONES = ("resnet18", "mobilenet_v2", "efficientnet_b0")
PRETRAINED_WEIGHTS = "IMAGENET1K_V1"  # ghi rõ để tái lập (cả 3 backbone đều có bản V1)


def _replace_head(model: nn.Module, name: str) -> None:
    """Thay lớp phân loại cuối bằng Linear(in_features, 1)."""
    if name == "resnet18":
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif name in ("mobilenet_v2", "efficientnet_b0"):
        # classifier = Sequential(Dropout, Linear): giữ Dropout, thay Linear
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    else:
        raise ValueError(f"Backbone chưa hỗ trợ: {name}")


def build_model(name: str, pretrained: bool = True) -> nn.Module:
    """Backbone đóng băng + head mới (chỉ head có requires_grad=True)."""
    if name not in SUPPORTED_BACKBONES:
        raise ValueError(f"Backbone '{name}' chưa hỗ trợ. Chọn một trong {SUPPORTED_BACKBONES}")

    model = get_model(name, weights=PRETRAINED_WEIGHTS if pretrained else None)
    for p in model.parameters():
        p.requires_grad = False
    _replace_head(model, name)  # tham số của head mới mặc định requires_grad=True
    return model


def set_head_only_train_mode(model: nn.Module) -> None:
    """model.train() nhưng giữ mọi lớp BatchNorm ở chế độ eval."""
    model.train()
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def head_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Các tham số có thể huấn luyện (chính là head). Dùng để lưu/khôi phục checkpoint nhẹ."""
    return {n: p.detach() for n, p in model.named_parameters() if p.requires_grad}


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Trả về (tổng số tham số, số tham số huấn luyện được)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@torch.no_grad()
def measure_latency_ms(
    model: nn.Module, img_size: int, device: torch.device, runs: int = 30, warmup: int = 5
) -> float:
    """Thời gian suy luận trung bình (ms) cho 1 ảnh, batch size 1, chỉ tính phần mô hình."""
    model.eval()
    x = torch.randn(1, 3, img_size, img_size, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / runs * 1000
