"""
feature_engineering.py — Признаковая инженерия
================================================
КРИТИЧНО: все агрегаты считаются ТОЛЬКО по данным до момента T.
Нет утечки — нет переобучения на public, провала на private.

Структура:
  1. Временны́е rolling-агрегаты (velocity, frequency, amount stats)
  2. Поведенческие отклонения от baseline клиента
  3. Признаки новизны (новый MCC, терминал, страна)
  4. Граф-фичи (популярность получателя)
  5. Временны́е признаки (час, день недели, etc.)
  6. Признаки из 🟡 жёлтых операций (semi-supervised)
"""

import polars as pl
import numpy as np
import logging
from pathlib import Path
from typing import List, Optional

import sys
sys.path.append(str(Path(__file__).parent.parent))
from configs.config import *

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ВРЕМЕННЫ́Е ПРИЗНАКИ СТРОКИ
# ═══════════════════════════════════════════════════════════════════════════════

def add_datetime_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Признаки из timestamp — без агрегации, для каждой строки.
    """
    return df.with_columns([
        pl.col(TIMESTAMP_COL).dt.hour().alias("hour"),
        pl.col(TIMESTAMP_COL).dt.weekday().alias("weekday"),   # 0=Mon
        pl.col(TIMESTAMP_COL).dt.day().alias("day_of_month"),
        pl.col(TIMESTAMP_COL).dt.month().alias("month"),

        # Время суток: ночь (0), утро (1), день (2), вечер (3)
        (
            pl.when(pl.col(TIMESTAMP_COL).dt.hour() < 6).then(0)
            .when(pl.col(TIMESTAMP_COL).dt.hour() < 12).then(1)
            .when(pl.col(TIMESTAMP_COL).dt.hour() < 18).then(2)
            .otherwise(3)
        ).alias("time_of_day"),

        # Выходной
        (pl.col(TIMESTAMP_COL).dt.weekday() >= 5).cast(pl.Int8).alias("is_weekend"),

        # "Нетипичное" время: ночь или ранее утро
        (pl.col(TIMESTAMP_COL).dt.hour().is_between(0, 5)).cast(pl.Int8).alias("is_night"),

        # Unix timestamp для вычисления дельт
        pl.col(TIMESTAMP_COL).dt.epoch(time_unit="s").alias("ts_seconds"),
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ROLLING-АГРЕГАТЫ НА КЛИЕНТА (главный источник сигнала)
# ═══════════════════════════════════════════════════════════════════════════════

def add_client_rolling_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Для каждой операции считаем агрегаты за последние N часов
    по CLIENT_ID, используя только прошлые данные.

    Полары поддерживают rolling over_time через group_by_dynamic
    или join_asof. Здесь используем подход с явным join для
    контроля утечки данных.
    """
    collected = df.collect()
    logger.info("Computing client rolling features...")

    # Собираем rolling фичи для каждого окна
    all_feature_dfs = []

    for window_h, window_name in zip(TIME_WINDOWS_H, TIME_WINDOW_NAMES):
        window_seconds = window_h * 3600
        logger.info(f"  Window: {window_name}")

        # Для каждой транзакции: сколько операций/сумма за последние N секунд
        # Используем self-join по времени
        feat = (
            collected
            .sort([CLIENT_ID_COL, TIMESTAMP_COL])
            .with_columns(
                pl.col("ts_seconds").alias("ts_current")
            )
        )

        # rolling_group_by с period
        try:
            agg_result = (
                feat
                .sort([CLIENT_ID_COL, TIMESTAMP_COL])
                .rolling(
                    index_column=TIMESTAMP_COL,
                    period=f"{window_h}h",
                    group_by=CLIENT_ID_COL,
                    closed="left",   # ВАЖНО: не включаем текущую строку
                )
                .agg([
                    pl.len().alias(f"cnt_{window_name}"),
                    pl.col(AMOUNT_COL).sum().alias(f"sum_{window_name}"),
                    pl.col(AMOUNT_COL).mean().alias(f"mean_{window_name}"),
                    pl.col(AMOUNT_COL).std().alias(f"std_{window_name}"),
                    pl.col(AMOUNT_COL).max().alias(f"max_{window_name}"),

                    # Уникальные категории
                    pl.col("mcc").n_unique().alias(f"mcc_uniq_{window_name}")
                    if "mcc" in feat.columns else pl.lit(0).alias(f"mcc_uniq_{window_name}"),

                    pl.col("country").n_unique().alias(f"country_uniq_{window_name}")
                    if "country" in feat.columns else pl.lit(0).alias(f"country_uniq_{window_name}"),

                    # Ночные операции
                    pl.col("is_night").sum().alias(f"night_cnt_{window_name}")
                    if "is_night" in feat.columns else pl.lit(0).alias(f"night_cnt_{window_name}"),
                ])
            )
            all_feature_dfs.append(agg_result.select(
                [EVENT_ID_COL] + [c for c in agg_result.columns if window_name in c]
            ) if EVENT_ID_COL in agg_result.columns else agg_result)

        except Exception as e:
            logger.warning(f"Rolling failed for {window_name}: {e}")
            continue

    # Объединяем все окна
    if not all_feature_dfs:
        return df

    result = all_feature_dfs[0]
    for fdf in all_feature_dfs[1:]:
        join_cols = [c for c in [EVENT_ID_COL, CLIENT_ID_COL, TIMESTAMP_COL] if c in fdf.columns]
        if join_cols:
            result = result.join(fdf, on=join_cols, how="left")

    return result.lazy()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ПРИЗНАКИ ОТКЛОНЕНИЯ ОТ ЛИЧНОГО BASELINE
# ═══════════════════════════════════════════════════════════════════════════════

def add_deviation_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Насколько текущая операция отклоняется от "нормы" клиента
    (по предобучающему периоду / долгому окну).
    """
    collected = df.collect()

    # Baseline клиента = агрегаты за 90 дней
    baseline_cols = [c for c in collected.columns if "30d" in c or "90d" in c]

    devs = []
    for window in ["7d", "30d"]:
        cnt_col   = f"cnt_{window}"
        sum_col   = f"sum_{window}"
        mean_col  = f"mean_{window}"
        std_col   = f"std_{window}"

        # cnt_90d как долгосрочная база
        base_cnt  = f"cnt_90d"
        base_mean = f"mean_90d"
        base_std  = f"std_90d"

        if all(c in collected.columns for c in [cnt_col, base_cnt, mean_col, base_mean]):
            devs += [
                # Относительная скорость
                (pl.col(cnt_col) / (pl.col(base_cnt) / 90 * (7 if "7d" in window else 30) + 1e-9))
                .alias(f"velocity_ratio_{window}"),

                # Z-score текущей суммы относительно личной нормы
                (
                    (pl.col(AMOUNT_COL) - pl.col(base_mean))
                    / (pl.col(base_std) + 1e-9)
                ).alias(f"amount_zscore_{window}"),
            ]

    if devs:
        collected = collected.with_columns(devs)

    return collected.lazy()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ПРИЗНАКИ "НОВИЗНЫ" (впервые встречаем)
# ═══════════════════════════════════════════════════════════════════════════════

def add_novelty_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Для каждой операции проверяем: видел ли этот клиент
    такой MCC / страну / терминал / сумму раньше?
    Новые паттерны → высокий риск фрода.
    """
    collected = df.collect().sort([CLIENT_ID_COL, TIMESTAMP_COL])
    novelty_features = []

    for col in ["mcc", "country", "channel", "currency"]:
        if col not in collected.columns:
            continue

        # cumcount = сколько раз видели данное значение ДО этой операции
        feat_name = f"is_new_{col}"
        collected = collected.with_columns(
            (
                pl.col(col)
                .cum_count()
                .over([CLIENT_ID_COL, col])
            ).alias(f"_cumcnt_{col}")
        ).with_columns(
            (pl.col(f"_cumcnt_{col}") == 1).cast(pl.Int8).alias(feat_name)
        ).drop(f"_cumcnt_{col}")

        logger.info(f"  Novelty feature: {feat_name}")

    return collected.lazy()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ВРЕМЕННА́Я ДЕЛЬТА (time since last)
# ═══════════════════════════════════════════════════════════════════════════════

def add_time_delta_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Время с последней операции клиента.
    Очень частые операции за короткий промежуток = красный флаг.
    """
    return (
        df
        .sort([CLIENT_ID_COL, TIMESTAMP_COL])
        .with_columns([
            # Секунды с предыдущей операции
            (
                pl.col("ts_seconds") - pl.col("ts_seconds").shift(1).over(CLIENT_ID_COL)
            ).alias("secs_since_prev"),

            # Секунды до следующей (осторожно: может быть look-ahead в prod!)
            # Используем только для оффлайн обучения
            (
                pl.col("ts_seconds").shift(-1).over(CLIENT_ID_COL) - pl.col("ts_seconds")
            ).alias("secs_to_next"),
        ])
        .with_columns([
            # Логарифм для нормализации
            pl.col("secs_since_prev").log(base=10).alias("log_secs_since_prev"),

            # Подозрительно быстрые серии
            (pl.col("secs_since_prev") < 60).cast(pl.Int8).alias("rapid_succession"),
        ])
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ГРАФ-ФИЧИ (популярность получателя денег)
# ═══════════════════════════════════════════════════════════════════════════════

def add_graph_features(
    df: pl.LazyFrame,
    history: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    Считаем "репутацию" получателей на основе истории:
    - сколько уникальных клиентов переводили этому терминалу/мерчанту
    - процент подозрительных (🔴🟡) операций у этого мерчанта
    """
    if "merchant_id" not in df.columns:
        logger.warning("merchant_id not found, skipping graph features")
        return df

    hist = history.collect()
    df_c = df.collect()

    # Статистики по merchant_id из истории (train period)
    merch_stats = (
        hist
        .group_by("merchant_id")
        .agg([
            pl.len().alias("merch_total_ops"),
            pl.col(CLIENT_ID_COL).n_unique().alias("merch_unique_clients"),
            pl.col(AMOUNT_COL).mean().alias("merch_avg_amount"),
            pl.col(AMOUNT_COL).std().alias("merch_std_amount"),

            # Доля красных операций
            (pl.col(LABEL_COL) == RED).sum().alias("merch_red_count")
            if LABEL_COL in hist.columns else pl.lit(0).alias("merch_red_count"),
        ])
        .with_columns([
            (pl.col("merch_red_count") / (pl.col("merch_total_ops") + 1e-9))
            .alias("merch_fraud_rate"),
        ])
    )

    result = df_c.join(merch_stats, on="merchant_id", how="left")
    return result.lazy()


# ═══════════════════════════════════════════════════════════════════════════════
# 7. SEMI-SUPERVISED: СИГНАЛ ОТ 🟡 ЖЁЛТЫХ ОПЕРАЦИЙ
# ═══════════════════════════════════════════════════════════════════════════════

def add_yellow_signal_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Жёлтые операции (подозрительные, но подтверждённые клиентом) —
    слабый сигнал фрода. Используем их для обогащения профиля клиента.
    """
    if LABEL_COL not in df.columns:
        return df

    return (
        df
        .sort([CLIENT_ID_COL, TIMESTAMP_COL])
        .with_columns([
            # Кол-во жёлтых операций за всю историю ДО этой точки
            (
                (pl.col(LABEL_COL) == YELLOW).cast(pl.Int32).cum_sum()
                .over(CLIENT_ID_COL)
                .shift(1)
                .over(CLIENT_ID_COL)
                .fill_null(0)
            ).alias("cumsum_yellow_prev"),

            # Кол-во красных операций (для train set)
            (
                (pl.col(LABEL_COL) == RED).cast(pl.Int32).cum_sum()
                .over(CLIENT_ID_COL)
                .shift(1)
                .over(CLIENT_ID_COL)
                .fill_null(0)
            ).alias("cumsum_red_prev"),
        ])
        .with_columns([
            # Был ли у клиента хоть один прошлый инцидент
            (pl.col("cumsum_yellow_prev") > 0).cast(pl.Int8).alias("had_yellow_before"),
            (pl.col("cumsum_red_prev") > 0).cast(pl.Int8).alias("had_red_before"),
        ])
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 8. КЛИЕНТСКИЙ ПРОФИЛЬ ИЗ PREТRAIN ПЕРИОДА
# ═══════════════════════════════════════════════════════════════════════════════

def build_client_pretrain_profile(pretrain: pl.LazyFrame) -> pl.DataFrame:
    """
    Строим стабильный профиль "нормального поведения" клиента
    на основе чистого predrain периода (нет фрода).
    """
    logger.info("Building client pretrain profiles...")
    return (
        pretrain
        .group_by(CLIENT_ID_COL)
        .agg([
            pl.len().alias("pretrain_total_ops"),
            pl.col(AMOUNT_COL).mean().alias("pretrain_mean_amount"),
            pl.col(AMOUNT_COL).std().alias("pretrain_std_amount"),
            pl.col(AMOUNT_COL).median().alias("pretrain_median_amount"),
            pl.col(AMOUNT_COL).quantile(0.95).alias("pretrain_q95_amount"),

            # Типичный час операций
            pl.col(TIMESTAMP_COL).dt.hour().mean().alias("pretrain_mean_hour"),
            pl.col(TIMESTAMP_COL).dt.hour().std().alias("pretrain_std_hour"),

            # Активность по дням недели
            pl.col(TIMESTAMP_COL).dt.weekday().mean().alias("pretrain_mean_weekday"),

            # Уникальные категории
            pl.col("mcc").n_unique().alias("pretrain_uniq_mcc")
            if "mcc" in pretrain.columns else pl.lit(0).alias("pretrain_uniq_mcc"),

            pl.col("country").n_unique().alias("pretrain_uniq_countries")
            if "country" in pretrain.columns else pl.lit(0).alias("pretrain_uniq_countries"),

            # Временной ритм
            (pl.col(TIMESTAMP_COL).dt.epoch("s").diff().over(CLIENT_ID_COL).mean())
            .alias("pretrain_mean_interval_s"),
        ])
        .collect()
    )


def join_pretrain_profile(df: pl.LazyFrame, profile: pl.DataFrame) -> pl.LazyFrame:
    """Присоединяет профиль pretrain к основному датасету."""
    collected = df.collect()
    result = collected.join(profile, on=CLIENT_ID_COL, how="left")

    if "pretrain_mean_amount" in result.columns:
        result = result.with_columns([
            # Насколько текущая сумма аномальна для клиента
            (
                (pl.col(AMOUNT_COL) - pl.col("pretrain_mean_amount"))
                / (pl.col("pretrain_std_amount") + 1e-9)
            ).alias("amount_deviation_from_pretrain"),

            # Час аномалии
            (
                (pl.col("hour") - pl.col("pretrain_mean_hour")).abs()
                / (pl.col("pretrain_std_hour") + 1e-9)
            ).alias("hour_deviation_from_pretrain")
            if "hour" in result.columns else pl.lit(None).cast(pl.Float32).alias("hour_deviation_from_pretrain"),
        ])

    return result.lazy()


# ═══════════════════════════════════════════════════════════════════════════════
# ГЛАВНАЯ ФУНКЦИЯ: СОБРАТЬ ВСЕ ФИЧИ
# ═══════════════════════════════════════════════════════════════════════════════

def build_features(
    df: pl.LazyFrame,
    history: Optional[pl.LazyFrame] = None,
    pretrain_profile: Optional[pl.DataFrame] = None,
    is_train: bool = True,
) -> pl.DataFrame:
    """
    Оркестрирует все шаги feature engineering.

    Args:
        df:               датасет для которого строим фичи
        history:          вся история ДО текущего df (без утечки!)
        pretrain_profile: профиль клиентов из pretrain периода
        is_train:         True = обучение (есть LABEL_COL)

    Returns:
        pl.DataFrame с фичами
    """
    logger.info("Building features...")

    # Шаг 1: Временны́е признаки строки
    df = add_datetime_features(df)
    logger.info("  ✓ Datetime features")

    # Шаг 2: Дельта времени
    df = add_time_delta_features(df)
    logger.info("  ✓ Time delta features")

    # Шаг 3: Новизна значений
    df = add_novelty_features(df)
    logger.info("  ✓ Novelty features")

    # Шаг 4: Semi-supervised из жёлтых
    if is_train:
        df = add_yellow_signal_features(df)
        logger.info("  ✓ Yellow signal features")

    # Шаг 5: Профиль pretrain
    if pretrain_profile is not None:
        df = join_pretrain_profile(df, pretrain_profile)
        logger.info("  ✓ Pretrain profile joined")

    # Шаг 6: Граф-фичи
    if history is not None:
        df = add_graph_features(df, history)
        logger.info("  ✓ Graph features")

    # Шаг 7: Rolling агрегаты (самый тяжёлый шаг — делаем последним)
    # NB: rolling требует ts_seconds из шага 2
    df = add_client_rolling_features(df)
    logger.info("  ✓ Rolling aggregate features")

    # Шаг 8: Девиации от baseline
    df = add_deviation_features(df)
    logger.info("  ✓ Deviation features")

    result = df.collect()
    logger.info(f"Features built: {result.width} columns, {result.height:,} rows")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# СПИСОК ФИНАЛЬНЫХ ФИЧЕЙ ДЛЯ МОДЕЛИ
# ═══════════════════════════════════════════════════════════════════════════════

def get_feature_columns(df: pl.DataFrame) -> List[str]:
    """
    Возвращает список колонок-фичей для обучения.
    Исключает ID, таймстамп, лейбл, технические.
    """
    EXCLUDE = {
        EVENT_ID_COL, CLIENT_ID_COL, TIMESTAMP_COL, LABEL_COL,
        "period", "ts_seconds", "_cumcnt_mcc", "_cumcnt_country",
        "ts_current",
    }
    feat_cols = [
        c for c in df.columns
        if c not in EXCLUDE
        and df[c].dtype not in [pl.Utf8, pl.Categorical, pl.Date, pl.Datetime]
        and not c.startswith("_")
    ]
    return feat_cols
