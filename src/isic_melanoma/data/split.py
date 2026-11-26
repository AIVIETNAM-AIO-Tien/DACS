"""Lấy mẫu và chia train/val/test THEO BỆNH NHÂN.

Quy trình:
1. Giữ toàn bộ ảnh ác tính, lấy ngẫu nhiên `n_benign` ảnh lành tính.
2. StratifiedGroupKFold (group = patient_id) lấy fold đầu làm test.
3. Trên phần còn lại, StratifiedGroupKFold lần nữa lấy fold đầu làm validation.
4. Kiểm tra không có bệnh nhân nào xuất hiện ở hai tập, rồi lưu CSV để tái lập.

    uv run python -m isic_melanoma.data.split [--force]
"""
from __future__ import annotations

import argparse

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from ..config import Config

REQUIRED_COLS = {"image_name", "patient_id", "target"}
SPLITS = ("train", "val", "test")


def sample_images(df: pd.DataFrame, n_benign: int, seed: int) -> pd.DataFrame:
    """Giữ toàn bộ ảnh ác tính + lấy mẫu ngẫu nhiên ảnh lành tính (ở mức ảnh)."""
    malignant = df[df["target"] == 1]
    benign = df[df["target"] == 0]
    benign = benign.sample(n=min(n_benign, len(benign)), random_state=seed)
    return pd.concat([malignant, benign]).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _hold_out_first_fold(df: pd.DataFrame, n_splits: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trả về (phần còn lại, fold đầu tiên) của StratifiedGroupKFold."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rest_idx, held_idx = next(iter(sgkf.split(df, df["target"], groups=df["patient_id"])))
    return df.iloc[rest_idx].reset_index(drop=True), df.iloc[held_idx].reset_index(drop=True)


def make_split(raw: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Tạo DataFrame có thêm cột `split` in {train, val, test}."""
    missing = REQUIRED_COLS - set(raw.columns)
    if missing:
        raise ValueError(f"train.csv thiếu cột: {sorted(missing)}")
    if raw["patient_id"].isna().any():
        raise ValueError("Có patient_id bị thiếu, không thể chia theo bệnh nhân.")

    df = sample_images(raw, cfg.n_benign, cfg.split_seed)
    train_val, test = _hold_out_first_fold(df, cfg.n_splits_test, cfg.split_seed)
    train, val = _hold_out_first_fold(train_val, cfg.n_splits_val, cfg.split_seed)

    return pd.concat(
        [train.assign(split="train"), val.assign(split="val"), test.assign(split="test")],
        ignore_index=True,
    )


def verify_split(df: pd.DataFrame) -> None:
    """Kiểm tra tính đúng đắn của split. Ném AssertionError nếu có vấn đề."""
    assert df["image_name"].is_unique, "image_name bị trùng"

    patients = {s: set(df.loc[df["split"] == s, "patient_id"]) for s in SPLITS}
    for s in SPLITS:
        assert len(patients[s]) > 0, f"Tập {s} rỗng"
        classes = set(df.loc[df["split"] == s, "target"])
        assert classes == {0, 1}, f"Tập {s} không đủ cả hai lớp: {classes}"

    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = patients[a] & patients[b]
        assert not overlap, f"Rò rỉ bệnh nhân giữa {a} và {b}: {len(overlap)} bệnh nhân"


def summarize_split(df: pd.DataFrame) -> pd.DataFrame:
    """Bảng thống kê: số ảnh, số bệnh nhân, số ca ác tính, tỉ lệ ác tính theo từng tập."""
    rows = []
    for s in SPLITS:
        part = df[df["split"] == s]
        n_mal = int((part["target"] == 1).sum())
        rows.append(
            {
                "split": s,
                "n_images": len(part),
                "n_patients": part["patient_id"].nunique(),
                "n_malignant": n_mal,
                "malignant_pct": round(100 * n_mal / len(part), 2),
            }
        )
    return pd.DataFrame(rows)


def load_or_create_split(cfg: Config, force: bool = False) -> pd.DataFrame:
    """Đọc split đã lưu; nếu chưa có (hoặc force=True) thì tạo mới, kiểm tra và lưu."""
    if cfg.split_csv.exists() and not force:
        df = pd.read_csv(cfg.split_csv)
        verify_split(df)
        return df

    cfg.ensure_dirs()
    raw = pd.read_csv(cfg.train_csv)
    df = make_split(raw, cfg)
    verify_split(df)
    df.to_csv(cfg.split_csv, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Tạo và kiểm tra split theo bệnh nhân")
    parser.add_argument("--force", action="store_true", help="Tạo lại split dù file đã tồn tại")
    args = parser.parse_args()

    cfg = Config()
    df = load_or_create_split(cfg, force=args.force)
    print(f"Split lưu tại: {cfg.split_csv}")
    print(summarize_split(df).to_string(index=False))
    print("Kiểm tra: không có bệnh nhân nào xuất hiện ở hai tập.")


if __name__ == "__main__":
    main()
