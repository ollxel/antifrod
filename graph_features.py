"""
graph_features.py
Графовые признаки: клиенты + получатели образуют bipartite-граф.
Фродовые кольца часто видны как кластеры в этом графе.
"""

import polars as pl
import numpy as np
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def build_transaction_graph(df: pl.DataFrame, cfg: dict) -> dict:
    """
    Строит словари смежности для графа client → merchant.
    Возвращает:
      client_out_degree: {client_id: число уникальных получателей}
      merchant_in_degree: {merchant_id: число уникальных отправителей}
      merchant_fraud_neighbors: {merchant_id: доля клиентов с фродом}
    """
    cid   = cfg["columns"]["client_id"]
    mid   = "merchant_id"

    if mid not in df.columns:
        return {}

    logger.info("Building transaction graph ...")

    # Степени
    client_out = (
        df.group_by(cid)
        .agg(pl.col(mid).n_unique().alias("client_out_degree"))
    )

    merchant_in = (
        df.group_by(mid)
        .agg(pl.col(cid).n_unique().alias("merchant_in_degree"))
    )

    result = {
        "client_out_degree": dict(zip(
            client_out[cid].to_list(),
            client_out["client_out_degree"].to_list()
        )),
        "merchant_in_degree": dict(zip(
            merchant_in[mid].to_list(),
            merchant_in["merchant_in_degree"].to_list()
        )),
    }

    # PageRank-приближение: мерчанты с высоким in-degree подозрительны
    if "label" in df.columns:
        fraud_by_merchant = (
            df
            .filter(pl.col("label").is_not_null())
            .group_by(mid)
            .agg([
                pl.col("label").mean().alias("merchant_fraud_neighbor_rate"),
                pl.col("label").sum().alias("merchant_fraud_count"),
            ])
        )
        result["fraud_by_merchant"] = fraud_by_merchant

    return result


def add_graph_features(
    target: pl.DataFrame,
    graph: dict,
    cfg: dict,
) -> pl.DataFrame:
    """Присоединяет графовые признаки к целевому DataFrame."""
    cid = cfg["columns"]["client_id"]
    mid = "merchant_id"

    logger.info("Adding graph features ...")
    result = target

    if "client_out_degree" in graph:
        mapping = graph["client_out_degree"]
        result = result.with_columns(
            pl.col(cid).replace(mapping, default=0).alias("client_out_degree")
        )

    if "merchant_in_degree" in graph and mid in result.columns:
        mapping = graph["merchant_in_degree"]
        result = result.with_columns(
            pl.col(mid).replace(mapping, default=0).alias("merchant_in_degree")
        )

    if "fraud_by_merchant" in graph and mid in result.columns:
        result = result.join(
            graph["fraud_by_merchant"], on=mid, how="left"
        )
        result = result.with_columns([
            pl.col("merchant_fraud_neighbor_rate").fill_null(0.0),
            pl.col("merchant_fraud_count").fill_null(0),
        ])

    return result


def shared_merchant_features(
    history: pl.DataFrame,
    target: pl.DataFrame,
    cfg: dict,
    top_k: int = 5,
) -> pl.DataFrame:
    """
    Для каждого клиента: сколько из его топ-получателей
    также являются получателями у клиентов с подтверждённым фродом.
    Это ключевой признак для обнаружения «мошеннических колец».
    """
    cid = cfg["columns"]["client_id"]
    mid = "merchant_id"

    if mid not in history.columns or "label" not in history.columns:
        return target

    logger.info("Computing shared merchant features ...")

    # Мерчанты, которые получали деньги от фродовых клиентов
    fraud_merchants = (
        history
        .filter(pl.col("label") == 1)
        .select(mid)
        .unique()
    )
    fraud_merchant_set = set(fraud_merchants[mid].to_list())

    # Для каждого клиента: доля его мерчантов в «чёрном списке»
    client_merchants = (
        history
        .group_by(cid)
        .agg(pl.col(mid).unique().alias("merchants"))
    )

    def fraud_merchant_overlap(merchants):
        if not merchants:
            return 0.0
        overlap = sum(1 for m in merchants if m in fraud_merchant_set)
        return overlap / len(merchants)

    client_merchants = client_merchants.with_columns(
        pl.col("merchants")
        .map_elements(fraud_merchant_overlap, return_dtype=pl.Float32)
        .alias("fraud_merchant_overlap")
    ).drop("merchants")

    return target.join(client_merchants, on=cid, how="left")
