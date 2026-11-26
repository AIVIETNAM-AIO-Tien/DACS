"""
    from .config import Config
    cfg = Config()                         
    cfg2 = replace(cfg, img_size=160)       # dataclasses.replace cho ablation
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    data_dir: Path = PROJECT_ROOT / "data" / "raw"    
    splits_dir: Path = PROJECT_ROOT / "data" / "splits" 
    results_dir: Path = PROJECT_ROOT / "results"         

    img_size: int = 224
    n_benign: int = 5000      # số ảnh lành tính lấy mẫu
    split_seed: int = 42      
    n_splits_test: int = 7    #  1/7 bệnh nhân vào test
    n_splits_val: int = 6     #  1/6 số bệnh nhân còn lại vào validation


    batch_size: int = 32
    num_workers: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 15
    patience: int = 3                # early stopping theo selection_metric trên validation
    use_class_weight: bool = True    

 
    backbones: tuple[str, ...] = ("resnet18", "mobilenet_v2", "efficientnet_b0")
    train_seeds: tuple[int, ...] = (0, 1, 2)   
    selection_metric: str = "val_auc"          # chọn epoch tốt nhất và backbone tốt nhất


    @property
    def train_csv(self) -> Path:
        return self.data_dir / "train.csv"

    @property
    def jpeg_dir(self) -> Path:
        return self.data_dir / "jpeg" / "train"

    @property
    def split_csv(self) -> Path:
        return self.splits_dir / f"split_seed{self.split_seed}_nb{self.n_benign}.csv"

    @property
    def benchmark_csv(self) -> Path:
        return self.results_dir / "benchmark.csv"

    def ensure_dirs(self) -> None:
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
