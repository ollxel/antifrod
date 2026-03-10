"""
eda.py — Разведочный анализ данных
====================================
Запускать: python notebooks/eda.py
Или в Jupyter: jupyter nbconvert --to notebook --execute eda.py

Секции:
  1. Базовая статистика по периодам
  2. Распределение целевых классов
  3. Паттерны по времени (час, день)
  4. Топ MCC и каналов у фрода
  5. Распределение сумм
  6. Корреляция с таргетом
  7. Аномальные клиенты
"""

import polars as pl
import polars.selectors as cs
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from configs.config import *
from src.data_loader import load_all_periods, describe_dataset


def section(title: str) -> None:
    print(f"\n{'▓'*60}")
    print(f"  {title}")
    print(f"{'▓'*60}")


# ─── Загрузка ─────────────────────────────────────────────────────────────────
section("1. Загрузка данных")
pretrain, train, pretest, test = load_all_periods(include_pretrain=True, lazy=False)

describe_dataset(train,   "TRAIN   (2024-10 → 2025-05)")
describe_dataset(pretest, "PRETEST (2025-06 → 2025-08)")
describe_dataset(test,    "TEST    (финальный день)")


# ─── Распределение классов ────────────────────────────────────────────────────
section("2. Распределение целевых классов")

if LABEL_COL in train.columns:
    dist = (
        train
        .group_by(LABEL_COL)
        .agg([
            pl.len().alias("count"),
            pl.col(AMOUNT_COL).mean().alias("avg_amount"),
            pl.col(AMOUNT_COL).median().alias("med_amount"),
        ])
        .sort(LABEL_COL)
    )
    total = dist["count"].sum()
    print(dist.with_columns(
        (pl.col("count") / total * 100).round(2).alias("pct%")
    ))


# ─── Паттерны по времени ──────────────────────────────────────────────────────
section("3. Паттерны фрода по времени суток")

if LABEL_COL in train.columns:
    hourly = (
        train
        .with_columns(pl.col(TIMESTAMP_COL).dt.hour().alias("hour"))
        .group_by(["hour", LABEL_COL])
        .agg(pl.len().alias("count"))
        .sort(["hour", LABEL_COL])
        .pivot(on=LABEL_COL, index="hour", values="count", aggregate_function="first")
        .sort("hour")
    )
    print("Операции по часам (по классам):")
    print(hourly)

    weekday = (
        train
        .with_columns(pl.col(TIMESTAMP_COL).dt.weekday().alias("weekday"))
        .group_by(["weekday", LABEL_COL])
        .agg(pl.len().alias("count"))
        .sort(["weekday", LABEL_COL])
    )
    print("\nОперации по дням недели:")
    print(weekday)


# ─── Топ MCC у фрода ─────────────────────────────────────────────────────────
section("4. Топ MCC (merchant category codes) у фрода vs норма")

if "mcc" in train.columns and LABEL_COL in train.columns:
    red_ops   = train.filter(pl.col(LABEL_COL) == RED)
    green_ops = train.filter(pl.col(LABEL_COL) == GREEN)

    red_mcc   = red_ops.group_by("mcc").agg(pl.len().alias("red_count")).sort("red_count", descending=True)
    green_mcc = green_ops.group_by("mcc").agg(pl.len().alias("green_count"))

    combined = (
        red_mcc.join(green_mcc, on="mcc", how="left")
        .with_columns(
            (pl.col("red_count") / (pl.col("green_count") + 1)).alias("fraud_ratio")
        )
        .sort("fraud_ratio", descending=True)
    )
    print("Top-20 MCC по fraud_ratio:")
    print(combined.head(20))


# ─── Распределение сумм ───────────────────────────────────────────────────────
section("5. Распределение сумм по классам")

if LABEL_COL in train.columns:
    amount_stats = (
        train
        .group_by(LABEL_COL)
        .agg([
            pl.col(AMOUNT_COL).min().alias("min"),
            pl.col(AMOUNT_COL).quantile(0.25).alias("q25"),
            pl.col(AMOUNT_COL).median().alias("median"),
            pl.col(AMOUNT_COL).quantile(0.75).alias("q75"),
            pl.col(AMOUNT_COL).quantile(0.95).alias("q95"),
            pl.col(AMOUNT_COL).max().alias("max"),
            pl.col(AMOUNT_COL).mean().alias("mean"),
        ])
        .sort(LABEL_COL)
    )
    print(amount_stats)


# ─── Velocity (скорость накопления) ──────────────────────────────────────────
section("6. Velocity — кол-во операций за 1 час до фрод-операции")

if LABEL_COL in train.columns:
    sorted_train = train.sort([CLIENT_ID_COL, TIMESTAMP_COL])

    # Для каждой операции: сколько операций этого клиента за предыдущий час
    velocity = (
        sorted_train
        .with_columns(
            pl.col(TIMESTAMP_COL).dt.epoch("s").alias("ts_s")
        )
        .with_columns(
            pl.col("ts_s").rolling_sum(window_size=3600, min_periods=0)
            .over(CLIENT_ID_COL)
            .alias("ops_in_1h")   # приближение
        )
    )
    print("Velocity по классам:")
    print(
        velocity.group_by(LABEL_COL)
        .agg(pl.col("ops_in_1h").mean().alias("avg_ops_1h"))
        .sort(LABEL_COL)
    )


# ─── Новые получатели ─────────────────────────────────────────────────────────
section("7. Доля операций с новым получателем/страной по классам")

if "country" in train.columns and LABEL_COL in train.columns:
    with_novelty = (
        train.sort([CLIENT_ID_COL, TIMESTAMP_COL])
        .with_columns(
            (pl.col("country").cum_count().over([CLIENT_ID_COL, "country"]) == 1)
            .cast(pl.Int8).alias("is_new_country")
        )
    )
    print(
        with_novelty
        .group_by(LABEL_COL)
        .agg(pl.col("is_new_country").mean().alias("pct_new_country"))
        .sort(LABEL_COL)
    )


# ─── Аномальные клиенты ───────────────────────────────────────────────────────
section("8. Топ-10 клиентов по количеству фрод-операций")

if LABEL_COL in train.columns:
    top_fraud_clients = (
        train
        .filter(pl.col(LABEL_COL) == RED)
        .group_by(CLIENT_ID_COL)
        .agg([
            pl.len().alias("red_ops"),
            pl.col(AMOUNT_COL).sum().alias("total_fraud_amount"),
        ])
        .sort("red_ops", descending=True)
        .head(10)
    )
    print(top_fraud_clients)


# ─── Дрейф данных между периодами ────────────────────────────────────────────
section("9. Сравнение распределений train vs pretest (drift check)")

if AMOUNT_COL in train.columns:
    train_stats   = train.select(pl.col(AMOUNT_COL).describe())
    pretest_stats = pretest.select(pl.col(AMOUNT_COL).describe())
    print(f"Train amount stats:\n{train_stats}")
    print(f"\nPretest amount stats:\n{pretest_stats}")

    # PSI (Population Stability Index) — простая версия
    def psi_score(reference: np.ndarray, test: np.ndarray, bins: int = 10) -> float:
        """PSI > 0.2 = значительный дрейф."""
        ref_counts, bin_edges = np.histogram(reference, bins=bins)
        test_counts, _        = np.histogram(test, bins=bin_edges)
        ref_pct  = (ref_counts + 1) / (len(reference) + bins)
        test_pct = (test_counts + 1) / (len(test) + bins)
        psi = np.sum((test_pct - ref_pct) * np.log(test_pct / ref_pct))
        return float(psi)

    psi = psi_score(
        train[AMOUNT_COL].drop_nulls().to_numpy(),
        pretest[AMOUNT_COL].drop_nulls().to_numpy(),
    )
    status = "✅ стабильно" if psi < 0.1 else ("⚠️ умеренный дрейф" if psi < 0.2 else "🔴 сильный дрейф")
    print(f"\nPSI для amount: {psi:.4f} → {status}")


print("\n\n✅ EDA завершён!")
