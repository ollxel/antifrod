"""
eda.py — Exploratory Data Analysis.
Запуск: python eda.py --config configs/config.yaml
"""

import argparse
import logging
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.utils import load_config, setup_logging
from src.data_loader import load_all, preprocess, build_target

logger = logging.getLogger(__name__)


def run_eda(cfg: dict):
    import polars as pl

    setup_logging()
    logger.info("Loading data for EDA ...")
    data = load_all(cfg)

    for name, df in data.items():
        df = preprocess(df, cfg)
        if "target" in df.columns:
            df = build_target(df)
        data[name] = df

    train_df = data.get("train")
    if train_df is None:
        logger.error("train.parquet not found")
        return

    # ── Основная статистика ───────────────────────
    print("\n" + "="*60)
    print("TRAIN DATASET OVERVIEW")
    print("="*60)
    print(f"Shape: {train_df.shape}")
    print(f"\nDtypes:\n{train_df.dtypes}")
    print(f"\nNull counts:\n{train_df.null_count()}")

    # ── Распределение классов ─────────────────────
    if "label" in train_df.columns:
        label_dist = train_df["label"].value_counts().sort("label")
        print(f"\nLabel distribution:\n{label_dist}")

        n_fraud = (train_df["label"] == 1).sum()
        n_total = len(train_df)
        print(f"\nFraud rate: {n_fraud}/{n_total} = {n_fraud/n_total:.4%}")

    if "target" in train_df.columns:
        target_dist = train_df["target"].value_counts().sort("target")
        print(f"\nTarget distribution (0=green, 1=yellow, 2=red):\n{target_dist}")

    # ── Временные тренды ──────────────────────────
    ts = cfg["columns"]["timestamp"]
    if ts in train_df.columns:
        print(f"\nTime range: {train_df[ts].min()} → {train_df[ts].max()}")

        # Ежемесячный фрод
        monthly = (
            train_df
            .with_columns(pl.col(ts).dt.month().alias("month"))
            .group_by("month")
            .agg([
                pl.count().alias("total"),
                pl.col("label").sum().alias("fraud_count"),
            ])
            .sort("month")
            .with_columns(
                (pl.col("fraud_count") / pl.col("total")).alias("fraud_rate")
            )
        )
        print(f"\nMonthly fraud rates:\n{monthly}")

    # ── Статистики по сумме ───────────────────────
    amt = cfg["columns"]["amount"]
    if amt in train_df.columns:
        print(f"\nAmount statistics:")
        print(train_df[amt].describe())

        if "label" in train_df.columns:
            fraud_amt = train_df.filter(pl.col("label") == 1)[amt].describe()
            legit_amt = train_df.filter(pl.col("label") == 0)[amt].describe()
            print(f"\nFraud amounts:\n{fraud_amt}")
            print(f"\nLegit amounts:\n{legit_amt}")

    # ── Топ-10 подозрительных MCC ─────────────────
    if "mcc" in train_df.columns and "label" in train_df.columns:
        top_mcc = (
            train_df
            .filter(pl.col("label").is_not_null())
            .group_by("mcc")
            .agg([
                pl.count().alias("count"),
                pl.col("label").mean().alias("fraud_rate"),
            ])
            .filter(pl.col("count") > 100)
            .sort("fraud_rate", descending=True)
            .head(10)
        )
        print(f"\nTop-10 MCC by fraud rate (min 100 txns):\n{top_mcc}")

    # ── Анализ паттернов по времени суток ─────────
    if "hour" in train_df.columns and "label" in train_df.columns:
        hourly = (
            train_df
            .filter(pl.col("label").is_not_null())
            .group_by("hour")
            .agg(pl.col("label").mean().alias("fraud_rate"))
            .sort("hour")
        )
        print(f"\nHourly fraud rates:\n{hourly}")

    # ── Клиенты с наибольшим количеством фрода ────
    cid = cfg["columns"]["client_id"]
    if "label" in train_df.columns:
        top_fraud_clients = (
            train_df
            .filter(pl.col("label") == 1)
            .group_by(cid)
            .count()
            .sort("count", descending=True)
            .head(10)
        )
        print(f"\nTop-10 clients by fraud count:\n{top_fraud_clients}")

    # ── Корреляция признаков с таргетом ──────────
    num_cols = [
        c for c in train_df.columns
        if train_df[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
        and c not in {"label", "target", "event_id", cid}
    ]
    if "label" in train_df.columns and num_cols:
        label_arr = train_df["label"].fill_null(0).to_numpy()
        correlations = {}
        for col in num_cols:
            try:
                col_arr = train_df[col].fill_null(0).fill_nan(0).to_numpy()
                corr = np.corrcoef(col_arr, label_arr)[0, 1]
                if not np.isnan(corr):
                    correlations[col] = abs(corr)
            except Exception:
                pass

        top_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:20]
        print("\nTop-20 features by |correlation| with fraud label:")
        for feat, corr in top_corr:
            print(f"  {feat:40s}: {corr:.4f}")

    print("\n✅ EDA complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_eda(cfg)
