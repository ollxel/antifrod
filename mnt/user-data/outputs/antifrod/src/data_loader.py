"""
data_loader.py — Загрузка и первичная обработка данных
=======================================================
Использует Polars для эффективной работы с 200M+ строками.
Основной принцип: НИКАКОГО look-ahead — только данные до момента T.
"""

import polars as pl
import polars.selectors as cs
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple

import sys
sys.path.append(str(Path(__file__).parent.parent))
from configs.config import *

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ЗАГРУЗКА СЫРЫХ ДАННЫХ
# ═══════════════════════════════════════════════════════════════════════════════

def load_raw(path: Path, lazy: bool = True) -> pl.DataFrame | pl.LazyFrame:
    """
    Загружает parquet/csv файл.
    lazy=True → LazyFrame (рекомендуется для больших файлов).
    """
    path = Path(path)
    logger.info(f"Loading {path.name}...")

    if path.suffix == ".parquet":
        if lazy:
            return pl.scan_parquet(path)
        return pl.read_parquet(path)
    elif path.suffix == ".csv":
        if lazy:
            return pl.scan_csv(path)
        return pl.read_csv(path)
    else:
        raise ValueError(f"Unknown format: {path.suffix}")


def cast_dtypes(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Приводит типы данных к оптимальным для экономии памяти.
    """
    return (
        df
        .with_columns([
            # timestamp → datetime
            pl.col(TIMESTAMP_COL).cast(pl.Datetime("us")).alias(TIMESTAMP_COL),
            # amount → float32
            pl.col(AMOUNT_COL).cast(pl.Float32),
            # категориальные → category (enum)
            *[pl.col(c).cast(pl.Categorical) for c in CAT_COLS if c in df.columns],
            # label если есть
            pl.col(LABEL_COL).cast(pl.Int8) if LABEL_COL in df.columns else pl.lit(None).cast(pl.Int8).alias(LABEL_COL),
        ])
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ЧТЕНИЕ И ОБЪЕДИНЕНИЕ ПЕРИОДОВ
# ═══════════════════════════════════════════════════════════════════════════════

def load_all_periods(
    include_pretrain: bool = True,
    lazy: bool = True
) -> Tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    """
    Загружает все 4 периода данных.
    Возвращает: (pretrain, train, pretest, test)
    """
    pretrain = (
        load_raw(PRETRAIN_PATH, lazy=lazy)
        .pipe(cast_dtypes)
        .with_columns(pl.lit("pretrain").alias("period"))
    ) if include_pretrain and PRETRAIN_PATH.exists() else None

    train = (
        load_raw(TRAIN_PATH, lazy=lazy)
        .pipe(cast_dtypes)
        .with_columns(pl.lit("train").alias("period"))
    )

    pretest = (
        load_raw(PRETEST_PATH, lazy=lazy)
        .pipe(cast_dtypes)
        .with_columns(pl.lit("pretest").alias("period"))
    )

    test = (
        load_raw(TEST_PATH, lazy=lazy)
        .pipe(cast_dtypes)
        .with_columns(pl.lit("test").alias("period"))
    )

    return pretrain, train, pretest, test


def build_full_history(
    pretrain: Optional[pl.LazyFrame],
    train: pl.LazyFrame,
    pretest: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    Объединяет все данные в хронологическом порядке для
    построения rolling-фичей без утечки данных.
    """
    parts = [p for p in [pretrain, train, pretest] if p is not None]
    combined = pl.concat(parts, how="diagonal_relaxed")
    return combined.sort([CLIENT_ID_COL, TIMESTAMP_COL])


# ═══════════════════════════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНЫЕ УТИЛИТЫ
# ═══════════════════════════════════════════════════════════════════════════════

def get_class_distribution(df: pl.LazyFrame) -> dict:
    """Подсчёт распределения классов."""
    counts = (
        df
        .filter(pl.col(LABEL_COL).is_not_null())
        .group_by(LABEL_COL)
        .agg(pl.len().alias("count"))
        .collect()
    )
    return dict(zip(counts[LABEL_COL].to_list(), counts["count"].to_list()))


def filter_period(df: pl.LazyFrame, start: str, end: str) -> pl.LazyFrame:
    """Фильтр по временному периоду."""
    return df.filter(
        (pl.col(TIMESTAMP_COL) >= pl.lit(start).str.to_datetime("%Y-%m-%d"))
        & (pl.col(TIMESTAMP_COL) <= pl.lit(end).str.to_datetime("%Y-%m-%d") + pl.duration(days=1))
    )


def get_client_sample(df: pl.LazyFrame, n: int = 1000, seed: int = 42) -> pl.LazyFrame:
    """Берёт случайную выборку клиентов для отладки."""
    clients = (
        df
        .select(CLIENT_ID_COL)
        .unique()
        .collect()
        .sample(n=min(n, df.select(CLIENT_ID_COL).unique().collect().height), seed=seed)
    )
    return df.filter(pl.col(CLIENT_ID_COL).is_in(clients[CLIENT_ID_COL]))


def describe_dataset(df: pl.LazyFrame, name: str = "dataset") -> None:
    """Быстрое описание датасета."""
    collected = df.collect()
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Строк:          {collected.height:>15,}")
    print(f"  Колонок:        {collected.width:>15,}")
    print(f"  Клиентов:       {collected[CLIENT_ID_COL].n_unique():>15,}")
    if TIMESTAMP_COL in collected.columns:
        print(f"  Период:         {collected[TIMESTAMP_COL].min()} → {collected[TIMESTAMP_COL].max()}")
    if LABEL_COL in collected.columns and collected[LABEL_COL].is_not_null().any():
        dist = collected.group_by(LABEL_COL).agg(pl.len()).sort(LABEL_COL)
        print(f"  Разметка:")
        for row in dist.iter_rows(named=True):
            label_name = {0: "🟢 Green", 1: "🔴 Red (target)", 2: "🟡 Yellow"}.get(row[LABEL_COL], "?")
            print(f"    {label_name}: {row['len']:>10,}")
    print(f"{'='*60}\n")
    print(collected.describe())


# ═══════════════════════════════════════════════════════════════════════════════
# БЫСТРЫЙ EDA
# ═══════════════════════════════════════════════════════════════════════════════

def quick_eda(df: pl.LazyFrame, name: str = "train") -> None:
    """EDA: пропуски, уникальность, примеры аномалий."""
    collected = df.collect()
    print(f"\n📊 EDA: {name}")

    # Пропуски
    nulls = {c: collected[c].null_count() for c in collected.columns if collected[c].null_count() > 0}
    if nulls:
        print("\n  Пропуски:")
        for col, cnt in sorted(nulls.items(), key=lambda x: -x[1]):
            pct = cnt / collected.height * 100
            print(f"    {col:<30} {cnt:>10,} ({pct:.1f}%)")

    # Уникальность
    print("\n  Уникальность:")
    for col in CAT_COLS:
        if col in collected.columns:
            print(f"    {col:<30} {collected[col].n_unique():>10,}")

    # Amount статистики по классам
    if LABEL_COL in collected.columns:
        print("\n  Amount по классам:")
        stats = (
            collected
            .group_by(LABEL_COL)
            .agg([
                pl.col(AMOUNT_COL).mean().alias("mean"),
                pl.col(AMOUNT_COL).median().alias("median"),
                pl.col(AMOUNT_COL).max().alias("max"),
            ])
            .sort(LABEL_COL)
        )
        print(stats)
