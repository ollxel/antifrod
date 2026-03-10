# 🛡️ AntiFrod — Bank Transaction Fraud Detection

Полный ML-пайплайн для задачи Anti-Fraud классификации транзакций.

## Структура проекта

```
antifrod/
├── configs/
│   └── config.yaml              ← Все гиперпараметры и пути к данным
├── src/
│   ├── data_loader.py           ← Загрузка + базовый preprocessing (Polars)
│   ├── feature_engineering.py  ← Feature engineering: агрегаты, поведение, target encoding
│   ├── graph_features.py        ← Графовые признаки: степени вершин, мошеннические кольца
│   ├── sequence_features.py     ← GRU-эмбеддинги истории транзакций (PyTorch)
│   ├── models.py                ← LightGBM / CatBoost / XGBoost / MLP
│   ├── validation.py            ← Temporal CV без leakage, PR-AUC
│   ├── stacking.py              ← Rank average, Optuna-blending, Logistic stacking
│   └── utils.py                 ← Конфиг, логи, seed, сохранение сабмита
├── train.py                     ← Главный скрипт обучения
├── predict.py                   ← Генерация submission.csv
├── eda.py                       ← Разведочный анализ данных
└── requirements.txt
```

---

## Быстрый старт

### 1. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 2. Подготовка данных
Положи файлы в папку `data/`:
```
data/
├── pretrain.parquet   ← 2023-10-01 → 2024-09-30
├── train.parquet      ← 2024-10-01 → 2025-05-31  (с разметкой)
├── pretest.parquet    ← 2025-06-01 → 2025-08-09  (без разметки)
└── test.parquet       ← финальные операции для предсказания
```

### 3. Настройка конфига
Отредактируй `configs/config.yaml` — укажи правильные названия колонок:
```yaml
columns:
  event_id: "event_id"
  client_id: "client_id"
  timestamp: "event_time"    # ← название колонки с датой
  amount: "amount"           # ← название колонки с суммой
  target: "target"           # ← raw статус: 0=green, 1=yellow, 2=red
  merchant_id: "merchant_id"
  device_id: "device_id"
  mcc: "mcc"
```

### 4. EDA
```bash
python eda.py
```

### 5. Обучение
```bash
# Полный пайплайн
python train.py --config configs/config.yaml

# Быстрая отладка на подвыборке
python train.py --config configs/config.yaml --debug
```

### 6. Генерация сабмита
```bash
python predict.py --config configs/config.yaml --output submission.csv
```

---

## Архитектура решения

### Feature Engineering (без look-ahead leakage!)

| Тип признаков | Описание |
|---|---|
| **Скользящие агрегаты** | count, sum, mean, std за 1ч/6ч/24ч/7д/30д по клиенту |
| **Velocity** | отношение суммы за 1ч к сумме за 24ч (ускорение трат) |
| **Поведенческий профиль** | z-score суммы, отклонение от среднего часа, сравнение с медианой |
| **Новизна** | флаги нового мерчанта / устройства / MCC / валюты для клиента |
| **Сессионные** | время с последней операции, порядковый номер, разрыв > 30 дней |
| **Target encoding** | доля фрода по merchant_id, по MCC-коду |
| **Жёлтый сигнал** | доля жёлтых операций у клиента в истории (PU-learning) |
| **Графовые** | out-degree клиента, in-degree мерчанта, перекрытие с фрод-мерчантами |
| **GRU эмбеддинги** | 32-мерный вектор из истории последних 50 операций |

### Валидация (Temporal CV)

```
История   ──────────────────────────────────────────►
Pre-train | ← Fold 1 train → | gap | ← val 1 → |
          | ← Fold 2 train ──────→ | gap | ← val 2 → |
          | ← Fold 3 train ──────────────→ | gap | ← val 3 → |
          ...
```

Gap = 7 дней между train и validation — предотвращает утечку данных.

### Ансамблирование

```
LightGBM ──┐
CatBoost ──┤──→ Rank Normalize ──→ [Optuna Blend / Rank Average] ──→ submit
XGBoost  ──┤
GRU      ──┘
```

---

## Ключевые особенности

### Работа с дисбалансом классов
- `scale_pos_weight=20` в LightGBM/XGBoost
- `class_weights=[1, 20]` в CatBoost
- `BCEWithLogitsLoss(pos_weight=20)` в MLP/GRU
- Focal Loss в sequence model

### Использование жёлтых операций (PU-learning)
Жёлтые операции — это подтверждённые подозрительные операции.
Они НЕ являются целевым классом, но их наличие у клиента —
сильный сигнал. Признак `yellow_rate` учитывает это.

### Временная корректность
Все агрегаты и энкодинги считаются по данным СТРОГО ДО текущей операции
(параметр `closed="left"` в rolling functions, `shift(1)` для предыдущих значений).

---

## Метрика
```python
from sklearn.metrics import average_precision_score
score = average_precision_score(y_true, y_pred)
```
PR-AUC (Average Precision) — чувствительна к ранжированию редкого класса.

---

## Советы по улучшению

1. **Pseudo-labeling**: предсказать жёлтые как фрод с порогом > 0.8, добавить в обучение
2. **Adversarial validation**: проверить, насколько test отличается от train по распределению
3. **Feature selection**: убрать признаки с нулевой важностью (экономит память и время)
4. **Calibration**: PlattScaling / IsotonicRegression для калибровки вероятностей
5. **Semi-supervised**: обучить autoencoder на нормальных операциях, аномалии = фрод
