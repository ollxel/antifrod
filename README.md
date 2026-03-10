# 🛡️ Anti-Fraud Competition Pipeline

Решение задачи классификации мошеннических банковских операций.  
Метрика: **PR-AUC** (sklearn `average_precision_score`).

---

## 📁 Структура проекта

```
antifrod/
├── configs/
│   └── config.py          ← Все константы, пути, гиперпараметры
├── src/
│   ├── data_loader.py     ← Загрузка данных (Polars, 200M+ строк)
│   ├── feature_engineering.py  ← Признаковая инженерия (CORE)
│   ├── validation.py      ← Временна́я кросс-валидация (NO LEAKAGE)
│   ├── models.py          ← LGB / CatBoost / XGB / MLP / GRU
│   ├── ensemble.py        ← Stacking, rank averaging, Optuna weights
│   └── tuning.py          ← HPO через Optuna
├── notebooks/
│   └── eda.py             ← Разведочный анализ данных
├── data/                  ← Сюда кладёшь скачанные файлы
│   ├── pretrain.parquet
│   ├── train.parquet
│   ├── pretest.parquet
│   └── test.parquet
├── outputs/               ← Генерируется автоматически
│   ├── models/            ← Сохранённые модели
│   ├── features/          ← Кэш признаков
│   └── submission_*.csv   ← Файлы для отправки
├── main.py                ← Главный pipeline
└── requirements.txt
```

---

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Подготовка данных

Скачай файлы с соревнования и положи в `data/`:
```
data/pretrain.parquet    (или .csv)
data/train.parquet
data/pretest.parquet
data/test.parquet
```

> **Уточни** имена колонок в `configs/config.py` — особенно  
> `EVENT_ID_COL`, `CLIENT_ID_COL`, `TIMESTAMP_COL`, `AMOUNT_COL`, `LABEL_COL`.

### 3. Запуск

```bash
# Быстрый тест на 10k клиентах (~5 мин)
python main.py --mode fast

# Полный pipeline: все модели + HPO + stacking (~несколько часов)
python main.py --mode full

# Только HPO (Optuna)
python main.py --mode tune

# Только инференс (модели уже обучены)
python main.py --mode predict
```

### 4. EDA

```bash
python notebooks/eda.py
```

---

## 🧠 Архитектура решения

### Feature Engineering (самое важное!)

```
Временны́е признаки строки:
  hour, weekday, is_night, is_weekend

Rolling агрегаты по клиенту (БЕЗ look-ahead!):
  cnt_{1h/6h/1d/7d/30d/90d}
  sum_{...}, mean_{...}, std_{...}, max_{...}
  mcc_uniq_{...}, country_uniq_{...}
  night_cnt_{...}

Поведенческие отклонения:
  velocity_ratio_{7d/30d}          — скорость vs норма
  amount_zscore_{7d/30d}           — Z-score суммы
  amount_deviation_from_pretrain   — отклонение от личного baseline

Признаки новизны:
  is_new_mcc, is_new_country       — впервые у этого клиента
  is_new_channel, is_new_currency

Временны́е дельты:
  secs_since_prev, rapid_succession

Граф-признаки:
  merch_total_ops, merch_fraud_rate

Semi-supervised (из 🟡 жёлтых):
  cumsum_yellow_prev, had_yellow_before
  cumsum_red_prev,   had_red_before

Профиль из pretrain:
  pretrain_mean_amount, pretrain_std_amount
  pretrain_mean_hour, pretrain_uniq_mcc
```

### Модели

| Модель | Параметры | Особенности |
|--------|-----------|-------------|
| LightGBM | num_leaves=255, Focal-like via scale_pos_weight | Главная модель |
| CatBoost | depth=8, auto_class_weights=Balanced | Хорошо с категориями |
| XGBoost  | max_depth=8, tree_method=hist | Диверсификация |
| MLP      | 512→256→128, Focal Loss, BatchNorm | Нелинейные взаимодействия |
| GRU      | seq_len=50, hidden=128 | История как последовательность |

### Ансамбль

```
Уровень 0:  LGB + CatBoost + XGB + MLP
            ↓ OOF предсказания
Уровень 1:  Optuna оптимизация весов
            ИЛИ Logistic Regression (stacking)
            ИЛИ Rank Averaging
```

### Валидация (КРИТИЧНО — без утечки)

```
Pretrain: 2023-10 → 2024-09  ← строим профиль клиента
Train:    2024-10 → 2025-05  ← обучение + CV

  Fold 1: train≤2025-02, val=2025-03
  Fold 2: train≤2025-03, val=2025-04
  Fold 3: train≤2025-04, val=2025-05

Pretest:  2025-06 → 2025-08  ← история для тест-фичей
Test:     финальный день      ← предсказание
```

**Правило**: rolling агрегат для строки T использует только данные с `closed="left"`.

---

## ⚙️ Настройка под данные

### Если колонки называются по-другому

Открой `configs/config.py` и измени:
```python
EVENT_ID_COL  = "event_id"       # ← имя колонки с ID операции
CLIENT_ID_COL = "client_id"      # ← имя колонки с ID клиента  
TIMESTAMP_COL = "event_time"     # ← имя колонки с timestamp
AMOUNT_COL    = "amount"         # ← сумма операции
LABEL_COL     = "label"          # ← метка (0=green, 1=red, 2=yellow)

CAT_COLS = ["mcc", "channel", "currency", ...]  # ← все категориальные
```

### Если данные в CSV

```python
# data_loader.py автоматически определяет по расширению
PRETRAIN_PATH = DATA_DIR / "pretrain.csv"
```

### Scale_pos_weight для LGB

```python
# configs/config.py
# Подбирается как: len(green) / len(red)
# При 51k red и ~200M green → ~3900
# Но слишком большое значение → много false positives
# Рекомендую начинать с 30-100 и смотреть на val PR-AUC
"scale_pos_weight": 50,
```

---

## 📊 Ожидаемые результаты

| Конфигурация | PR-AUC (приблизительно) |
|-------------|------------------------|
| LGB baseline (только временны́е фичи) | ~0.15–0.20 |
| LGB + rolling агрегаты | ~0.25–0.35 |
| LGB + все фичи | ~0.35–0.45 |
| Ансамбль всех моделей | ~0.45–0.55 |

> Числа приблизительные и зависят от данных.

---

## 🔍 Ключевые идеи

1. **Pretrain период** = "чистая" история для baseline клиента
2. **Yellow (🟡)** = слабый сигнал; используем как semi-supervised сигнал, не как таргет
3. **Temporal CV** = валидация на будущем, обучение на прошлом (как в реальной системе)
4. **Rank averaging** = робастный ансамбль когда модели дают разные масштабы

---

## 📤 Submission

После запуска в `outputs/` появятся:
- `submission_weighted.csv` — взвешенный ансамбль (основной)
- `submission_rank.csv` — rank averaging (запасной)
- `submission_stacking.csv` — stacking мета-модель

Формат:
```
event_id,predict
125854726334416,-0.338988
125949211749418,-4.100378
```

Отправляй тот файл, у которого лучший PR-AUC на local validation!
