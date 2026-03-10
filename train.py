"""
train.py — Главный скрипт обучения.

Запуск:
    python train.py --config configs/config.yaml [--debug]
"""

import argparse
import logging
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import load_config, setup_logging, seed_everything, timer, save_predictions
from src.data_loader import load_all, preprocess, build_target, concat_history, downcast
from src.feature_engineering import build_features, get_feature_columns
from src.graph_features import build_transaction_graph, add_graph_features, shared_merchant_features
from src.validation import run_oof, pr_auc, TemporalSplit
from src.models import train_lgbm, train_catboost, train_xgboost, FraudMLP, get_feature_importance
from src.stacking import ensemble
from src.sequence_features import build_sequences, train_gru_encoder, get_gru_predictions

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════
# Обёртки для OOF runner
# ══════════════════════════════════════════════════

def make_lgbm_fn(cfg, feature_names):
    def fn(X_tr, y_tr, X_val, y_val, fold_idx):
        return train_lgbm(X_tr, y_tr, X_val, y_val, fold_idx, cfg, feature_names)
    return fn

def make_catboost_fn(cfg):
    def fn(X_tr, y_tr, X_val, y_val, fold_idx):
        return train_catboost(X_tr, y_tr, X_val, y_val, fold_idx, cfg)
    return fn

def make_xgb_fn(cfg):
    def fn(X_tr, y_tr, X_val, y_val, fold_idx):
        return train_xgboost(X_tr, y_tr, X_val, y_val, fold_idx, cfg)
    return fn


# ══════════════════════════════════════════════════
# Пайплайн
# ══════════════════════════════════════════════════

def main(args):
    # ── Инициализация ──────────────────────────────
    cfg = load_config(args.config)
    setup_logging(log_file="outputs/train.log")
    seed_everything(cfg["random_seed"])

    Path(cfg["data"]["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["data"]["cache_dir"]).mkdir(parents=True, exist_ok=True)

    if args.debug:
        logger.info("DEBUG MODE: using small subset of data")

    # ── 1. Загрузка данных ─────────────────────────
    with timer("Data loading"):
        data = load_all(cfg)

        # Preprocessing
        for name in data:
            data[name] = preprocess(data[name], cfg)
            if "target" in data[name].columns:
                data[name] = build_target(data[name])

        pretrain = data.get("pretrain")
        train_df = data.get("train")
        test_df  = data.get("test")
        pretest  = data.get("pretest")

        if train_df is None:
            raise FileNotFoundError("train.parquet not found!")

        if args.debug:
            # Берём первые 100k строк для отладки
            train_df = train_df.head(100_000)
            if pretrain is not None:
                pretrain = pretrain.head(200_000)

    # ── 2. Feature engineering ─────────────────────
    with timer("Feature engineering (train)"):
        # История для обучения: pretrain + train (кроме самих операций)
        if pretrain is not None:
            history_for_train = concat_history(pretrain, train_df)
        else:
            history_for_train = train_df

        train_features = build_features(
            history=history_for_train,
            target=train_df,
            cfg=cfg,
            cache_path=f"{cfg['data']['cache_dir']}/train_features.parquet",
        )

    # Графовые признаки
    if cfg["feature_engineering"]["use_graph_features"]:
        with timer("Graph features (train)"):
            graph = build_transaction_graph(history_for_train, cfg)
            train_features = add_graph_features(train_features, graph, cfg)
            train_features = shared_merchant_features(history_for_train, train_features, cfg)

    # ── 3. Подготовка матриц ───────────────────────
    feature_cols = get_feature_columns(train_features, cfg)
    logger.info(f"Feature count: {len(feature_cols)}")
    logger.info(f"Top features: {feature_cols[:20]}")

    # Только размеченные операции (красные + зелёные, без жёлтых для обучения)
    # Вариант 1: только красные (1) vs зелёные (0)
    # Вариант 2: красные (1) vs всё остальное
    labeled_mask = train_features["label"].is_not_null()
    train_labeled = train_features.filter(labeled_mask)

    import polars as pl
    X = train_labeled.select(feature_cols).to_numpy().astype(np.float32)
    y = train_labeled["label"].to_numpy().astype(np.float32)
    ts = train_labeled[cfg["columns"]["timestamp"]].to_numpy()

    logger.info(f"Training set: {X.shape}, fraud rate: {y.mean():.4f}")

    # ── 4. OOF обучение ────────────────────────────
    oof_preds = {}
    all_models = {}

    # LightGBM
    if cfg["models"]["lgbm"]["enabled"]:
        with timer("LightGBM OOF"):
            lgbm_oof, lgbm_models = run_oof(
                make_lgbm_fn(cfg, feature_cols),
                X, y, ts, cfg,
            )
            oof_preds["lgbm"] = lgbm_oof
            all_models["lgbm"] = lgbm_models

            imp = get_feature_importance(lgbm_models, feature_cols, "lgbm")
            logger.info("Top-20 LGB features:")
            for i, (f, v) in enumerate(list(imp.items())[:20]):
                logger.info(f"  {i+1:2d}. {f}: {v:.1f}")

    # CatBoost
    if cfg["models"]["catboost"]["enabled"]:
        with timer("CatBoost OOF"):
            cat_oof, cat_models = run_oof(
                make_catboost_fn(cfg),
                X, y, ts, cfg,
            )
            oof_preds["catboost"] = cat_oof
            all_models["catboost"] = cat_models

    # XGBoost
    if cfg["models"]["xgboost"]["enabled"]:
        with timer("XGBoost OOF"):
            xgb_oof, xgb_models = run_oof(
                make_xgb_fn(cfg),
                X, y, ts, cfg,
            )
            oof_preds["xgboost"] = xgb_oof
            all_models["xgboost"] = xgb_models

    # MLP
    if cfg["models"]["mlp"]["enabled"]:
        with timer("MLP OOF"):
            splitter = TemporalSplit(
                n_folds=cfg["validation"]["n_folds"],
                gap_days=cfg["validation"]["gap_days"],
            )
            import polars as pl
            tmp = pl.DataFrame({"ts": ts})
            folds = list(splitter.split(tmp, "ts"))

            mlp_oof = np.zeros(len(y))
            for fold_idx, (tr_idx, val_idx) in enumerate(folds):
                mlp = FraudMLP(cfg, input_dim=X.shape[1])
                mlp.fit(X[tr_idx], y[tr_idx], X[val_idx], y[val_idx])
                mlp_oof[val_idx] = mlp.predict(X[val_idx])
            oof_preds["mlp"] = mlp_oof

    # GRU sequence model
    if cfg["feature_engineering"]["use_sequence_features"]:
        with timer("GRU sequence model OOF"):
            try:
                seqs, event_ids_seq = build_sequences(
                    train_labeled,
                    cfg,
                    max_len=cfg["feature_engineering"]["sequence_max_len"],
                )
                # Сопоставляем порядок с X/y (по event_id)
                gru_model = train_gru_encoder(seqs, y, cfg)
                gru_oof = get_gru_predictions(gru_model, seqs)
                oof_preds["gru"] = gru_oof
                all_models["gru"] = [gru_model]
            except Exception as e:
                logger.warning(f"GRU training failed: {e}")

    # ── 5. OOF метрика ────────────────────────────
    logger.info("\n" + "="*50)
    logger.info("OOF Scores:")
    for name, preds in oof_preds.items():
        filled = preds != 0
        if filled.sum() > 0:
            score = pr_auc(y[filled], preds[filled])
            logger.info(f"  {name:12s}: {score:.5f}")
    logger.info("="*50)

    # ── 6. Feature engineering для теста ──────────
    if test_df is not None:
        with timer("Feature engineering (test)"):
            if pretest is not None and pretrain is not None:
                history_for_test = concat_history(pretrain, train_df, pretest)
            elif pretrain is not None:
                history_for_test = concat_history(pretrain, train_df)
            else:
                history_for_test = train_df

            test_features = build_features(
                history=history_for_test,
                target=test_df,
                cfg=cfg,
                cache_path=f"{cfg['data']['cache_dir']}/test_features.parquet",
            )

        if cfg["feature_engineering"]["use_graph_features"]:
            graph_test = build_transaction_graph(history_for_test, cfg)
            test_features = add_graph_features(test_features, graph_test, cfg)
            test_features = shared_merchant_features(history_for_test, test_features, cfg)

        X_test = test_features.select(feature_cols).to_numpy().astype(np.float32)
        test_event_ids = test_features[cfg["columns"]["event_id"]].to_list()

        # ── 7. Предсказания на тесте ──────────────
        with timer("Test predictions"):
            test_preds = {}

            # Переобучаем на всех данных (для теста — весь train)
            if "lgbm" in all_models:
                import lightgbm as lgb
                params = dict(cfg["models"]["lgbm"]["params"])
                params.pop("early_stopping_rounds", None)
                params["n_estimators"] = max(
                    m.best_iteration for m in all_models["lgbm"]
                    if hasattr(m, "best_iteration")
                ) if all_models["lgbm"] else 1000
                dtrain = lgb.Dataset(X, y)
                final_lgbm = lgb.train(params, dtrain)
                test_preds["lgbm"] = final_lgbm.predict(X_test)

            if "catboost" in all_models:
                from catboost import CatBoostClassifier, Pool
                params = dict(cfg["models"]["catboost"]["params"])
                params["iterations"] = np.mean([
                    m.best_iteration_ for m in all_models["catboost"]
                    if hasattr(m, "best_iteration_")
                ]) if all_models["catboost"] else 1000
                final_cat = CatBoostClassifier(**params)
                final_cat.fit(Pool(X, y))
                test_preds["catboost"] = final_cat.predict_proba(X_test)[:, 1]

            if "xgboost" in all_models:
                import xgboost as xgb
                params = dict(cfg["models"]["xgboost"]["params"])
                params.pop("early_stopping_rounds", None)
                params.pop("n_estimators", None)
                dtrain = xgb.DMatrix(X, label=y)
                dtest  = xgb.DMatrix(X_test)
                final_xgb = xgb.train(params, dtrain, num_boost_round=1500)
                test_preds["xgboost"] = final_xgb.predict(dtest)

            if "gru" in all_models and all_models["gru"]:
                seqs_test, _ = build_sequences(test_features, cfg)
                test_preds["gru"] = get_gru_predictions(all_models["gru"][0], seqs_test)

        # ── 8. Ансамблирование и сабмит ───────────
        with timer("Ensembling"):
            final_preds = ensemble(
                oof_preds={k: v for k, v in oof_preds.items() if k in test_preds},
                test_preds=test_preds,
                y_true=y,
                cfg=cfg,
            )

        save_predictions(
            event_ids=test_event_ids,
            predictions=final_preds,
            output_path=f"{cfg['data']['output_dir']}/submission.csv",
        )

    # Сохраняем OOF предсказания для анализа
    np.save(f"{cfg['data']['output_dir']}/oof_preds.npy", oof_preds)
    np.save(f"{cfg['data']['output_dir']}/oof_labels.npy", y)

    logger.info("\n✅ Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
