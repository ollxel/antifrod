"""
data_loader.py — Загрузка и первичная обработка данных
=======================================================
Использует Polars для эффективной работы с 200M+ строками.
"""

import polars as pl
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple

import sys
sys.path.append(str(Path(__file__).parent.parent))
from configs.config import (
    PRETRAIN_PATH, TRAIN_PATH, PRETEST_PATH, TEST_PATH,
    TIMESTAMP_COL, AMOUNT_COL, LABEL_COL, CLIENT_ID_COL, EVENT_ID_COL,
    CAT_COLS,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ЗАГРУЗКА СЫРЫХ ДАННЫХ
# ═══════════════════════════════════════════════════════════════════════════════

def load_raw(path: Path, lazy: bool = True) -> pl.DataFrame | pl.LazyFrame:
    """Загружает parquet/csv файл."""
    path = Path(path)
    logger.info(f"Loading {path.name}...")
    if path.suffix == ".parquet":
        return pl.scan_parquet(path) if lazy else pl.read_parquet(path)
    elif path.suffix == ".csv":
        return pl.scan_csv(path) if lazy else pl.read_csv(path)
    else:
        raise ValueError(f"Unknown format: {path.suffix}")


def cast_dtypes(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Приводит типы данных к оптимальным для экономии памяти.
    Использует collect_schema().names() вместо .columns
    чтобы избежать PerformanceWarning на LazyFrame.
    """
    schema_names = set(df.collect_schema().names())
    exprs = []

    if TIMESTAMP_COL in schema_names:
        # Используем strptime с явным форматом для парсинга временных строк
        exprs.append(pl.col(TIMESTAMP_COL).str.strptime(pl.Datetime("us"), format="%Y-%m-%d %H:%M:%S"))

    if AMOUNT_COL in schema_names:
        exprs.append(pl.col(AMOUNT_COL).cast(pl.Float32))

    for c in CAT_COLS:
        if c in schema_names:
            # Convert to string first to handle int/other types safely, then to categorical
            exprs.append(pl.col(c).cast(pl.Utf8).cast(pl.Categorical))

    if LABEL_COL in schema_names:
        exprs.append(pl.col(LABEL_COL).cast(pl.Int8))
    else:
        exprs.append(pl.lit(None).cast(pl.Int8).alias(LABEL_COL))

    return df.with_columns(exprs) if exprs else df


def _check_file(path: Path) -> None:
    """Выдаёт понятное сообщение если файл не найден."""
    if not path.exists():
        raise FileNotFoundError(
            f"\n\n  Файл не найден: {path}"
            f"\n  Положи файлы данных в папку: {path.parent}"
            f"\n  Ожидаемая структура:"
            f"\n    data/"
            f"\n    ├── pretrain.parquet"
            f"\n    ├── train.parquet"
            f"\n    ├── pretest.parquet"
            f"\n    └── test.parquet\n"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# ЧТЕНИЕ И ОБЪЕДИНЕНИЕ ПЕРИОДОВ
# ═══════════════════════════════════════════════════════════════════════════════

def load_all_periods(
    include_pretrain: bool = True,
    lazy: bool = True,
) -> Tuple[Optional[pl.LazyFrame], pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    """
    Загружает все 4 периода данных.
    Возвращает: (pretrain, train, pretest, test)
    pretrain может быть None если файл отсутствует.
    """
    for p in [TRAIN_PATH, PRETEST_PATH, TEST_PATH]:
        _check_file(p)

    pretrain = None
    if include_pretrain:
        if PRETRAIN_PATH.exists():
            pretrain = (
                load_raw(PRETRAIN_PATH, lazy=lazy)
                .pipe(cast_dtypes)
                .with_columns(pl.lit("pretrain").alias("period"))
            )
        else:
            logger.warning(f"pretrain.parquet not found at {PRETRAIN_PATH}, skipping.")

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

    logger.info(
        f"Loaded: pretrain={'yes' if pretrain is not None else 'no'}, "
        f"train=yes, pretest=yes, test=yes"
    )
    return pretrain, train, pretest, test


def build_full_history(
    pretrain: Optional[pl.LazyFrame],
    train: pl.LazyFrame,
    pretest: pl.LazyFrame,
) -> pl.LazyFrame:
    """Объединяет все данные в хронологическом порядке."""
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
    all_clients = df.select(pl.col(CLIENT_ID_COL).unique()).collect()
    n = min(n, all_clients.height)
    sampled = all_clients.sample(n=n, seed=seed)
    return df.filter(pl.col(CLIENT_ID_COL).is_in(sampled[CLIENT_ID_COL]))


def describe_dataset(df: pl.LazyFrame, name: str = "dataset") -> None:
    """Быстрое описание датасета."""
    collected = df.collect()
    label_map = {0: "Green", 1: "Red (target)", 2: "Yellow"}
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Строк:    {collected.height:>15,}")
    print(f"  Колонок:  {collected.width:>15,}")
    if CLIENT_ID_COL in collected.columns:
        print(f"  Клиентов: {collected[CLIENT_ID_COL].n_unique():>15,}")
    if TIMESTAMP_COL in collected.columns:
        print(f"  Период:   {collected[TIMESTAMP_COL].min()} → {collected[TIMESTAMP_COL].max()}")
    if LABEL_COL in collected.columns and collected[LABEL_COL].is_not_null().any():
        dist = collected.group_by(LABEL_COL).agg(pl.len()).sort(LABEL_COL)
        print(f"  Разметка:")
        for row in dist.iter_rows(named=True):
            lname = label_map.get(row[LABEL_COL], str(row[LABEL_COL]))
            print(f"    {lname}: {row['len']:>10,}")
    print(f"{'='*60}\n")


def quick_eda(df: pl.LazyFrame, name: str = "dataset") -> None:
    """EDA: пропуски, уникальность категорий."""
    collected = df.collect()
    print(f"\nEDA: {name}")

    nulls = {c: collected[c].null_count() for c in collected.columns if collected[c].null_count() > 0}
    if nulls:
        print("\n  Пропуски:")
        for col, cnt in sorted(nulls.items(), key=lambda x: -x[1]):
            print(f"    {col:<30} {cnt:>10,}  ({cnt/collected.height*100:.1f}%)")

    print("\n  Уникальные значения категорий:")
    for col in CAT_COLS:
        if col in collected.columns:
            print(f"    {col:<30} {collected[col].n_unique():>10,}")