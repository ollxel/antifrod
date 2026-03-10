"""
main.py — Главный pipeline соревнования
========================================
Запускай: python main.py [--mode full|fast|tune|predict]

Режимы:
  fast    — только LGB, без tuning (для быстрой итерации)
  full    — все модели + tuning + stacking
  tune    — только Optuna HPO
  predict — только генерация submission (модели уже обучены)
"""

import argparse
import logging
import time
import numpy as np
import polars as pl
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from configs.config import *
from src.data_loader import (
    load_all_periods, build_full_history,
    describe_dataset, quick_eda,
)
from src.feature_engineering import (
    build_features, get_feature_columns,
    build_client_pretrain_profile,
)
from src.models import (
    train_lgb_cv, train_lgb_full,
    train_catboost_cv, train_xgb_cv, train_mlp_cv,
    predict_lgb_ensemble, predict_cb_ensemble, predict_xgb_ensemble,
)
from src.ensemble import FinalEnsemble, rank_average
from src.validation import compute_pr_auc, build_oof_predictions

# ─── Логирование ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "pipeline.log"),
    ]
)
logger = logging.getLogger("main")


# ═══════════════════════════════════════════════════════════════════════════════
# ШАГИ PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def step_load_data(fast: bool = False):
    """Шаг 1: Загрузка данных."""
    logger.info("=" * 60)
    logger.info("STEP 1: Loading data")
    logger.info("=" * 60)

    pretrain, train, pretest, test = load_all_periods(include_pretrain=True)

    if fast:
        logger.info("FAST MODE: sampling 10k clients")
        from src.data_loader import get_client_sample
        train   = get_client_sample(train,   n=10_000)
        pretest = get_client_sample(pretest, n=10_000)
        test    = get_client_sample(test,    n=10_000)

    # EDA
    describe_dataset(train,   "TRAIN")
    describe_dataset(pretest, "PRETEST")
    describe_dataset(test,    "TEST")

    return pretrain, train, pretest, test


def step_feature_engineering(pretrain, train, pretest, test):
    """Шаг 2: Построение признаков."""
    logger.info("=" * 60)
    logger.info("STEP 2: Feature Engineering")
    logger.info("=" * 60)

    t0 = time.time()

    # Профиль клиента из pretrain
    pretrain_profile = None
    if pretrain is not None:
        logger.info("Building pretrain client profiles...")
        pretrain_profile = build_client_pretrain_profile(pretrain)
        pretrain_profile.write_parquet(FEATS_DIR / "pretrain_profile.parquet")
        logger.info(f"  Profile: {pretrain_profile.height:,} clients")

    # История для графовых фичей (pretrain + train без лейблов будущего)
    history_for_graph = build_full_history(pretrain, train, pretest)

    # Обучающие фичи
    logger.info("Building train features...")
    train_feats = build_features(
        train,
        history=pretrain,    # для train используем только pretrain как историю
        pretrain_profile=pretrain_profile,
        is_train=True,
    )
    train_feats.write_parquet(FEATS_DIR / "train_features.parquet")

    # Тестовые фичи (история = pretrain + train + pretest)
    logger.info("Building test features...")
    test_feats = build_features(
        test,
        history=history_for_graph,   # вся история до теста
        pretrain_profile=pretrain_profile,
        is_train=False,
    )
    test_feats.write_parquet(FEATS_DIR / "test_features.parquet")

    logger.info(f"Feature engineering done in {time.time()-t0:.1f}s")
    logger.info(f"  Train: {train_feats.shape}, Test: {test_feats.shape}")

    return train_feats, test_feats


def step_train_models(train_feats: pl.DataFrame, mode: str = "full"):
    """Шаг 3: Обучение всех моделей с CV."""
    logger.info("=" * 60)
    logger.info("STEP 3: Model Training")
    logger.info("=" * 60)

    feature_cols = get_feature_columns(train_feats)
    logger.info(f"Feature columns: {len(feature_cols)}")
    logger.info(f"  {feature_cols[:10]} ...")

    results = {}
    all_models = {}
    oof_preds = {}

    # ── LightGBM ──────────────────────────────────────────────────────────────
    logger.info("\n[LightGBM]")

    # Загружаем лучшие параметры если есть
    lgb_params_path = OUTPUT_DIR / "best_lgb_params.json"
    lgb_params = LGB_PARAMS.copy()
    if lgb_params_path.exists():
        import json
        with open(lgb_params_path) as f:
            saved = json.load(f)
        lgb_params.update({k: v for k, v in saved.items() if not k.startswith("_")})
        logger.info(f"  Loaded tuned LGB params from {lgb_params_path}")

    lgb_fold_results, lgb_models = train_lgb_cv(
        train_feats, feature_cols, params=lgb_params,
        use_yellow_as_negative=False,
    )
    results["lgb"] = lgb_fold_results
    all_models["lgb"] = lgb_models

    # OOF предсказания для stacking
    from src.validation import build_oof_predictions
    lgb_oof_preds, lgb_oof_labels = build_oof_predictions(lgb_fold_results)
    oof_preds["lgb"] = lgb_oof_preds
    y_oof = lgb_oof_labels   # одинаковый для всех моделей

    if mode == "full":
        # ── CatBoost ──────────────────────────────────────────────────────────
        logger.info("\n[CatBoost]")
        try:
            cb_fold_results, cb_models = train_catboost_cv(train_feats, feature_cols)
            results["catboost"] = cb_fold_results
            all_models["catboost"] = cb_models
            cb_oof, _ = build_oof_predictions(cb_fold_results)
            oof_preds["catboost"] = cb_oof
        except Exception as e:
            logger.error(f"CatBoost failed: {e}")

        # ── XGBoost ───────────────────────────────────────────────────────────
        logger.info("\n[XGBoost]")
        try:
            xgb_fold_results, xgb_models = train_xgb_cv(train_feats, feature_cols)
            results["xgb"] = xgb_fold_results
            all_models["xgb"] = xgb_models
            xgb_oof, _ = build_oof_predictions(xgb_fold_results)
            oof_preds["xgb"] = xgb_oof
        except Exception as e:
            logger.error(f"XGBoost failed: {e}")

        # ── MLP ───────────────────────────────────────────────────────────────
        logger.info("\n[MLP Neural Net]")
        try:
            mlp_fold_results, mlp_models = train_mlp_cv(train_feats, feature_cols)
            if mlp_fold_results:
                results["mlp"] = mlp_fold_results
                all_models["mlp"] = mlp_models
                mlp_oof, _ = build_oof_predictions(mlp_fold_results)
                oof_preds["mlp"] = mlp_oof
        except Exception as e:
            logger.error(f"MLP failed: {e}")

    return feature_cols, all_models, oof_preds, y_oof


def step_ensemble(oof_preds: dict, y_oof: np.ndarray):
    """Шаг 4: Оптимизация ансамбля."""
    logger.info("=" * 60)
    logger.info("STEP 4: Ensemble Optimization")
    logger.info("=" * 60)

    ensemble = FinalEnsemble()
    ensemble.fit_weights(oof_preds, y_oof)
    ensemble.evaluate_strategies(oof_preds, y_oof)
    return ensemble


def step_predict_and_submit(
    test_feats: pl.DataFrame,
    feature_cols: list,
    all_models: dict,
    ensemble: FinalEnsemble,
    strategy: str = "weighted",
):
    """Шаг 5: Предсказание на тесте + формирование submission."""
    logger.info("=" * 60)
    logger.info("STEP 5: Prediction & Submission")
    logger.info("=" * 60)

    X_test = test_feats.select(feature_cols).fill_nan(0).fill_null(0).to_numpy()
    event_ids = test_feats[EVENT_ID_COL].to_numpy()

    test_preds = {}

    # LGB
    if "lgb" in all_models:
        test_preds["lgb"] = predict_lgb_ensemble(all_models["lgb"], X_test)
        logger.info(f"  LGB preds: mean={test_preds['lgb'].mean():.4f}")

    # CatBoost
    if "catboost" in all_models:
        test_preds["catboost"] = predict_cb_ensemble(all_models["catboost"], X_test)

    # XGBoost
    if "xgb" in all_models:
        from src.models import predict_xgb_ensemble
        test_preds["xgb"] = predict_xgb_ensemble(all_models["xgb"], X_test, feature_cols)

    # Финальный ансамбль
    final_preds = ensemble.predict(test_preds, strategy=strategy)

    # Также делаем rank_average как запасной вариант
    rank_preds = rank_average(test_preds) if len(test_preds) > 1 else final_preds

    # Сохранение submission
    _save_submission(event_ids, final_preds, "submission_weighted.csv")
    _save_submission(event_ids, rank_preds,  "submission_rank.csv")
    # Stacking
    stacking_preds = ensemble.predict(test_preds, strategy="stacking")
    _save_submission(event_ids, stacking_preds, "submission_stacking.csv")

    logger.info(f"\nSubmission stats:")
    logger.info(f"  n_events:   {len(final_preds):,}")
    logger.info(f"  mean score: {final_preds.mean():.4f}")
    logger.info(f"  max score:  {final_preds.max():.4f}")
    logger.info(f"  pos>0.5:    {(final_preds > 0.5).sum():,}")

    return final_preds


def _save_submission(event_ids: np.ndarray, preds: np.ndarray, filename: str) -> None:
    """Сохраняет submission в правильном формате."""
    sub = pl.DataFrame({
        EVENT_ID_COL: event_ids,
        "predict":    preds,
    })
    path = OUTPUT_DIR / filename
    sub.write_csv(path)
    logger.info(f"  Saved: {path} ({len(preds):,} rows)")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Anti-Fraud Competition Pipeline")
    parser.add_argument(
        "--mode", default="fast",
        choices=["fast", "full", "tune", "predict"],
        help="Pipeline mode"
    )
    args = parser.parse_args()

    logger.info(f"Starting pipeline in mode: {args.mode.upper()}")
    logger.info(f"Output dir: {OUTPUT_DIR}")
    t_start = time.time()

    if args.mode == "tune":
        # Только Optuna HPO
        from src.tuning import tune_lightgbm, tune_catboost
        pretrain, train, pretest, test = step_load_data(fast=False)
        _, train_feats, _ = step_feature_engineering(pretrain, train, pretest, test)[:3]
        feature_cols = get_feature_columns(train_feats)
        logger.info("Tuning LightGBM...")
        tune_lightgbm(train_feats, feature_cols, n_trials=100)
        logger.info("Tuning CatBoost...")
        tune_catboost(train_feats, feature_cols, n_trials=50)
        return

    if args.mode == "predict":
        # Только инференс, модели уже обучены
        logger.info("Loading prebuilt models...")
        import lightgbm as lgb, json
        test_feats = pl.read_parquet(FEATS_DIR / "test_features.parquet")
        with open(FEATS_DIR / "feature_cols.json") as f:
            feature_cols = json.load(f)["cols"]

        models = [lgb.Booster(model_file=str(p)) for p in sorted(MODELS_DIR.glob("lgb_fold*.txt"))]
        X_test = test_feats.select(feature_cols).fill_nan(0).fill_null(0).to_numpy()
        preds  = predict_lgb_ensemble(models, X_test)
        _save_submission(test_feats[EVENT_ID_COL].to_numpy(), preds, "submission_predict.csv")
        return

    # ── Полный pipeline ───────────────────────────────────────────────────────
    fast = (args.mode == "fast")

    pretrain, train, pretest, test = step_load_data(fast=fast)
    train_feats, test_feats = step_feature_engineering(pretrain, train, pretest, test)

    # Сохраняем список фичей
    import json
    feature_cols_for_save = get_feature_columns(train_feats)
    with open(FEATS_DIR / "feature_cols.json", "w") as f:
        json.dump({"cols": feature_cols_for_save}, f)

    feature_cols, all_models, oof_preds, y_oof = step_train_models(
        train_feats, mode=args.mode
    )
    ensemble = step_ensemble(oof_preds, y_oof)
    step_predict_and_submit(test_feats, feature_cols, all_models, ensemble)

    logger.info(f"\n{'='*60}")
    logger.info(f"Pipeline complete in {(time.time()-t_start)/60:.1f} minutes")
    logger.info(f"Submissions saved to: {OUTPUT_DIR}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
