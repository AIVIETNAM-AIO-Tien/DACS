"""Vòng lặp benchmark: các backbone x các seed, ghi CSV, tổng hợp mean ± std, chọn backbone tốt nhất.

Đặt file này tại: src/<package_name>/benchmark.py

Chạy:
    uv run python -m <package_name>.benchmark
    uv run python -m <package_name>.benchmark --backbones resnet18 --seeds 0 1
    uv run python -m <package_name>.benchmark --force      # chạy lại, ghi đè dòng cũ

Có thể dừng và chạy lại: các cặp (backbone, seed) đã có trong benchmark.csv sẽ được bỏ qua.
Backbone tốt nhất được chọn theo trung bình `cfg.selection_metric` trên VALIDATION.
Cột test_* chỉ để báo cáo, không dùng để chọn.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .config import Config
from .train import train_one_run
from .utils import get_device, get_logger

SUMMARY_COLS = [
    "val_auc",
    "test_auc",
    "test_pr_auc",
    "test_sens_at_spec90",
    "test_sensitivity",
    "test_specificity",
    "latency_ms",
    "train_time_s",
    "params_trainable",
    "params_total",
]


def _upsert_row(path: Path, row: dict) -> None:
    """Thêm dòng vào CSV; nếu (backbone, seed) đã có thì thay thế."""
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        keep = ~((old["backbone"] == row["backbone"]) & (old["seed"] == row["seed"]))
        new = pd.concat([old[keep], new], ignore_index=True)
    new.to_csv(path, index=False)


def _done_pairs(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    return set(zip(df["backbone"], df["seed"].astype(int)))


def summarize(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Bảng mean/std theo backbone, sắp xếp giảm dần theo selection_metric (mean)."""
    cols = [c for c in SUMMARY_COLS if c in df.columns]
    summary = df.groupby("backbone")[cols].agg(["mean", "std"])
    summary.columns = [f"{c}_{stat}" for c, stat in summary.columns]
    summary["n_seeds"] = df.groupby("backbone").size()
    return summary.sort_values(f"{cfg.selection_metric}_mean", ascending=False)


def run_benchmark(
    cfg: Config | None = None,
    backbones: tuple[str, ...] | None = None,
    seeds: tuple[int, ...] | None = None,
    force: bool = False,
) -> pd.DataFrame:
    cfg = cfg or Config()
    backbones = backbones or cfg.backbones
    seeds = seeds or cfg.train_seeds
    cfg.ensure_dirs()

    logger = get_logger("benchmark", cfg.results_dir / "benchmark.log")
    device = get_device()
    logger.info(f"Thiết bị: {device} | backbones: {backbones} | seeds: {seeds}")

    done = set() if force else _done_pairs(cfg.benchmark_csv)
    for backbone in backbones:
        for seed in seeds:
            if (backbone, seed) in done:
                logger.info(f"Bỏ qua {backbone} seed {seed} (đã có kết quả)")
                continue
            row = train_one_run(cfg, backbone, seed, device=device, logger=logger)
            _upsert_row(cfg.benchmark_csv, row)  # ghi ngay sau mỗi lần chạy để không mất kết quả

    df = pd.read_csv(cfg.benchmark_csv)
    summary = summarize(df, cfg)
    summary.to_csv(cfg.results_dir / "benchmark_summary.csv")

    best = summary.index[0]
    (cfg.results_dir / "best_backbone.txt").write_text(best + "\n", encoding="utf-8")
    logger.info(f"Backbone tốt nhất theo {cfg.selection_metric} (trung bình): {best}")
    print(summary.round(4).to_string())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark các backbone (chỉ train head)")
    parser.add_argument("--backbones", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Chạy lại dù đã có kết quả")
    args = parser.parse_args()

    cfg = Config()
    if args.backbones:
        cfg = replace(cfg, backbones=tuple(args.backbones))
    if args.seeds:
        cfg = replace(cfg, train_seeds=tuple(args.seeds))
    run_benchmark(cfg, force=args.force)


if __name__ == "__main__":
    main()
