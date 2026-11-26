"""Dataset, transform và DataLoader

Ảnh gốc của cuộc thi rất lớn nên đọc và resize mỗi epoch sẽ chậm. Có thể tạo
một lần bộ ảnh đã resize (cache):
    uv run python -m <package_name>.data.dataset --build-cache
Nếu thư mục cache tồn tại, dataset sẽ tự đọc từ đó.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

from ..config import Config
from ..utils import make_generator, seed_worker
from .split import load_or_create_split

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Loaders(NamedTuple):
    train: DataLoader
    val: DataLoader
    test: DataLoader
    pos_weight: float  # n_âm / n_dương trên tập train, dùng cho BCEWithLogitsLoss


# ----------------------------------------------------------------------
# Đường dẫn ảnh và cache
# ----------------------------------------------------------------------
def cache_dir(cfg: Config) -> Path:
    return cfg.data_dir / "processed" / f"jpeg_{cfg.img_size}"


def get_image_dir(cfg: Config) -> Path:
    """Dùng ảnh đã resize nếu có, nếu không dùng ảnh gốc."""
    cache = cache_dir(cfg)
    return cache if cache.is_dir() else cfg.jpeg_dir


def build_resized_cache(cfg: Config) -> None:
    """Resize các ảnh nằm trong split về img_size x img_size và lưu một lần."""
    df = load_or_create_split(cfg)
    dst = cache_dir(cfg)
    dst.mkdir(parents=True, exist_ok=True)
    size = (cfg.img_size, cfg.img_size)

    for name in tqdm(df["image_name"], desc=f"Resize -> {dst.name}"):
        out_path = dst / f"{name}.jpg"
        if out_path.exists():
            continue
        with Image.open(cfg.jpeg_dir / f"{name}.jpg") as img:
            img.convert("RGB").resize(size, Image.Resampling.BILINEAR).save(out_path, quality=95)


def build_transforms(img_size: int, augment: bool = True) -> tuple[T.Compose, T.Compose]:
    """Trả về (train_transform, eval_transform).
    """
    normalize = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    eval_tf = T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), normalize])

    if not augment:
        return eval_tf, eval_tf

    train_tf = T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(20),
            T.ToTensor(),
            normalize,
        ]
    )
    return train_tf, eval_tf


class SkinLesionDataset(Dataset):
    """Trả về (ảnh đã transform, float32). Thứ tự theo `df`, giữ nguyên khi shuffle=False."""

    def __init__(self, df: pd.DataFrame, img_dir: Path, transform) -> None:
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.names = self.df["image_name"].tolist()
        self.labels = self.df["target"].astype("float32").tolist()

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, i: int):
        with Image.open(self.img_dir / f"{self.names[i]}.jpg") as img:
            image = self.transform(img.convert("RGB"))
        return image, torch.tensor(self.labels[i], dtype=torch.float32)


def make_dataloaders(cfg: Config, seed: int, augment: bool = True) -> Loaders:
    """Tạo loader cho train/val/test. `seed` điều khiển thứ tự shuffle của train."""
    split_df = load_or_create_split(cfg)
    img_dir = get_image_dir(cfg)
    train_tf, eval_tf = build_transforms(cfg.img_size, augment)

    datasets = {
        name: SkinLesionDataset(split_df[split_df["split"] == name], img_dir, tf)
        for name, tf in (("train", train_tf), ("val", eval_tf), ("test", eval_tf))
    }

    common = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        persistent_workers=cfg.num_workers > 0,
    )   
    train_loader = DataLoader(datasets["train"], shuffle=True, generator=make_generator(seed), **common)
    val_loader = DataLoader(datasets["val"], shuffle=False, **common)
    test_loader = DataLoader(datasets["test"], shuffle=False, **common)

    y = datasets["train"].df["target"]
    pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1))
    return Loaders(train_loader, val_loader, test_loader, pos_weight)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-cache", action="store_true", help="Resize ảnh một lần để huấn luyện nhanh hơn")
    args = parser.parse_args()
    if args.build_cache:
        build_resized_cache(Config())


if __name__ == "__main__":
    main()
