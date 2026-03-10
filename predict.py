"""
predict.py — Генерация сабмита по обученным моделям.
Запуск: python predict.py --config configs/config.yaml --models_dir outputs/
"""

import argparse
import logging
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.utils import load_config, setup_logging, save_predictions
from src.data_loader import load_all, preprocess
from src.feature_engineering import build_features, get_feature_columns
from src.graph_features import build_transaction_graph, add_graph_features, shared_merchant_features
from src.stacking import rank_normalize, rank_average

logger = logging.getLogger(__name__)


def predict(args):
    cfg = load_config(args.config)
    setup_logging()

    # ── Загрузка данных ─────────────────────────
    data = load_all(cfg)
    for name in data:
        data[name] = preprocess(data[name], cfg)

    pretrain = data.get("pretrain")
    train_df = data.get("train")
    pretest  = data.get("pretest")
    test_df  = data.get("test")

    if test_df is None:
        raise FileNotFoundError("test.parquet not found!")

    # ── История для фич ─────────────────────────
    from src.data_loader import concat_history
    parts = [p for p in [pretrain, train_df, pretest] if p is not None]
    history = concat_history(*parts) if len(parts) > 1 else parts[0]

    # ── Feature engineering ──────────────────────
    test_features = build_features(
        history=history,
        target=test_df,
        cfg=cfg,
        cache_path=f"{cfg['data']['cache_dir']}/test_features_predict.parquet",
    )

    if cfg["feature_engineering"]["use_graph_features"]:
        graph = build_transaction_graph(history, cfg)
        test_features = add_graph_features(test_features, graph, cfg)
        test_features = shared_merchant_features(history, test_features, cfg)

    feature_cols = get_feature_columns(test_features, cfg)
    X_test = test_features.select(feature_cols).to_numpy().astype(float)
    test_event_ids = test_features[cfg["columns"]["event_id"]].to_list()

    # ── Загрузка моделей и предсказание ──────────
    import pickle
    models_dir = Path(args.models_dir)
    test_preds = {}

    for model_type in ["lgbm", "catboost", "xgboost"]:
        model_files = sorted(models_dir.glob(f"{model_type}_fold*.pkl"))
        if not model_files:
            continue

        fold_preds = []
        for mf in model_files:
            with open(mf, "rb") as f:
                model = pickle.load(f)

            if model_type == "lgbm":
                import lightgbm as lgb
                pred = model.predict(X_test, num_iteration=model.best_iteration)
            elif model_type == "catboost":
                pred = model.predict_proba(X_test)[:, 1]
            elif model_type == "xgboost":
                import xgboost as xgb
                pred = model.predict(xgb.DMatrix(X_test))

            fold_preds.append(pred)

        test_preds[model_type] = np.mean([rank_normalize(p) for p in fold_preds], axis=0)
        logger.info(f"Loaded {len(fold_preds)} {model_type} models")

    if not test_preds:
        raise RuntimeError("No saved models found in " + str(models_dir))

    # ── Финальное ансамблирование ─────────────────
    final_preds = rank_average(test_preds)

    save_predictions(
        event_ids=test_event_ids,
        predictions=final_preds,
        output_path=args.output,
    )
    logger.info(f"✅ Submission saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--models_dir", default="outputs/")
    parser.add_argument("--output", default="outputs/submission.csv")
    args = parser.parse_args()
    predict(args)
