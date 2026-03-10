"""
Anti-Fraud Competition — Central Configuration
================================================
Все пути, гиперпараметры и константы в одном месте.
"""

from pathlib import Path
import os

# ─── Пути ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
DATA_DIR    = ROOT / "data"
OUTPUT_DIR  = ROOT / "outputs"
MODELS_DIR  = ROOT / "outputs" / "models"
FEATS_DIR   = ROOT / "outputs" / "features"
LOGS_DIR    = ROOT / "outputs" / "logs"

for d in [OUTPUT_DIR, MODELS_DIR, FEATS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Файлы данных ─────────────────────────────────────────────────────────────
PRETRAIN_PATH   = DATA_DIR / "pretrain.parquet"   # 2023-10 → 2024-09
TRAIN_PATH      = DATA_DIR / "train.parquet"      # 2024-10 → 2025-05
PRETEST_PATH    = DATA_DIR / "pretest.parquet"    # 2025-06 → 2025-08-09
TEST_PATH       = DATA_DIR / "test.parquet"       # финальный день
SAMPLE_SUB_PATH = DATA_DIR / "sample_submit.csv"

# ─── Временные периоды ────────────────────────────────────────────────────────
PRETRAIN_START  = "2023-10-01"
PRETRAIN_END    = "2024-09-30"
TRAIN_START     = "2024-10-01"
TRAIN_END       = "2025-05-31"
PRETEST_START   = "2025-06-01"
TEST_END        = "2025-08-09"

# ─── Схема данных (реальные колонки датасета) ────────────────────────────────
EVENT_ID_COL    = "event_id"
CLIENT_ID_COL   = "customer_id"           # реальное название
TIMESTAMP_COL   = "event_dttm"            # реальное название
AMOUNT_COL      = "operaton_amt"          # опечатка в данных — именно так
LABEL_COL       = "label"                 # 0=green, 1=red(target), 2=yellow

# Категориальные колонки
CAT_COLS = [
    "event_type_nm",
    "channel_indicator_type",
    "channel_indicator_sub_type",
    "currency_iso_cd",
    "mcc_code",
    "pos_cd",
    "operating_system_type",
]

# Числовые колонки транзакции
NUM_COLS = [
    "operaton_amt",
    "battery",
]

# ─── Метки классов ────────────────────────────────────────────────────────────
GREEN  = 0   # подтверждена / нет обратной связи
RED    = 1   # не подтверждена (ЦЕЛЕВОЙ КЛАСС 🔴)
YELLOW = 2   # подозрительная, но подтверждена 🟡

# ─── Временны́е окна для агрегатов (в часах) ──────────────────────────────────
TIME_WINDOWS_H = [1, 6, 24, 24*7, 24*30, 24*90]   # 1ч, 6ч, 1д, 7д, 30д, 90д
TIME_WINDOW_NAMES = ["1h", "6h", "1d", "7d", "30d", "90d"]

# ─── LightGBM параметры ───────────────────────────────────────────────────────
LGB_PARAMS = {
    "objective":        "binary",
    "metric":           "average_precision",
    "boosting_type":    "gbdt",
    "num_leaves":       255,
    "max_depth":        -1,
    "learning_rate":    0.05,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "min_child_samples": 50,
    "lambda_l1":        0.1,
    "lambda_l2":        1.0,
    "scale_pos_weight": 30,   # ~отношение neg/pos; подбирать
    "n_jobs":           -1,
    "verbose":          -1,
    "seed":             42,
}
LGB_N_ROUNDS      = 3000
LGB_EARLY_STOP    = 100
SCALE_POS_WEIGHT  = 30   # ~отношение neg/pos; подбирать через tuning

# ─── CatBoost параметры ───────────────────────────────────────────────────────
CB_PARAMS = {
    "iterations":       2000,
    "learning_rate":    0.05,
    "depth":            8,
    "loss_function":    "Logloss",
    "eval_metric":      "AUC",
    "random_seed":      42,
    "auto_class_weights": "Balanced",
    "task_type":        "GPU",   # поменяй на CPU если нет GPU
    "verbose":          200,
}

# ─── Кросс-валидация (временна́я) ─────────────────────────────────────────────
# Разбиваем TRAIN период на фолды по времени
# Fold 1: train до 2025-03, val 2025-04–2025-05
# Fold 2: train до 2025-02, val 2025-03–2025-04
# Fold 3: train до 2025-01, val 2025-02–2025-03
CV_FOLDS = [
    {"train_end": "2025-02-28", "val_start": "2025-03-01", "val_end": "2025-03-31"},
    {"train_end": "2025-03-31", "val_start": "2025-04-01", "val_end": "2025-04-30"},
    {"train_end": "2025-04-30", "val_start": "2025-05-01", "val_end": "2025-05-31"},
]

# ─── Ансамблирование ──────────────────────────────────────────────────────────
ENSEMBLE_WEIGHTS = {
    "lgb":      0.40,
    "catboost": 0.30,
    "xgb":      0.20,
    "nn":       0.10,
}

SEED = 42