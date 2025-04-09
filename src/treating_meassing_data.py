import numpy as np
import pandas as pd

def treating_meassing_data(df):
    # -------------------------
    # Valores globais auxiliares
    # -------------------------
    median_amount_global = df["tx_amount"].median()
    
    # -------------------------
    # Rolling stats
    # -------------------------
    std_cols = [col for col in df.columns if col.startswith("std_amount_per_customer_last_")]
    df[std_cols] = df[std_cols].fillna(1e-6)

    # Substitui zscore por 0 (neutro)
    zscore_cols = [col for col in df.columns if col.startswith("amount_zscore_per_customer_last_")]
    df[zscore_cols] = df[zscore_cols].fillna(0)

    # -------------------------
    # Tempo entre transações
    # -------------------------
    if 'time_since_last_tx' in df.columns:
        df["time_since_last_tx"] = df["time_since_last_tx"].fillna(df["time_since_last_tx"].max())

    if 'mean_time_between_txs' in df.columns:
        df["mean_time_between_txs"] = df["mean_time_between_txs"].fillna(df["mean_time_between_txs"].median())

    if 'std_time_between_txs' in df.columns:
        df["std_time_between_txs"] = df["std_time_between_txs"].fillna(df["std_time_between_txs"].median())

    # -------------------------
    # Média e desvio padrão por cliente
    # -------------------------
    if 'std_amount' in df.columns:
        df["std_amount"] = df["std_amount"].fillna(1e-6)

    if 'mean_amount' in df.columns:
        df["mean_amount"] = df["mean_amount"].fillna(df["mean_amount"].median())
        
    if 'tx_amount_to_mean_ration' in df.columns:
        df["tx_amount_to_mean_ration"] = df["tx_amount_to_mean_ration"].fillna(
            df["tx_amount"] / df["mean_amount"]
        )

    median_amount_global = df["tx_amount"].median()
    if 'tx_amount_median_ratio' in df.columns:
        df["tx_amount_median_ratio"] = df["tx_amount_median_ratio"].fillna(
            df["tx_amount"] / median_amount_global
        )
        
    # -------------------------
    # Razões entre janelas
    # -------------------------
    ratio_cols = [col for col in df.columns if col.startswith("ratio_mean_amount_per_customer_last_1h_to_")]
    df[ratio_cols] = df[ratio_cols].fillna(1.0)

    # -------------------------
    # Padrões por hora do dia
    # -------------------------
    if "nb_tx_hour" in df.columns and "tx_hour" in df.columns:
        fallback_count = df.groupby("tx_hour")["nb_tx_hour"].median()
        df["nb_tx_hour"] = df.apply(
            lambda row: fallback_count[row["tx_hour"]] if pd.isna(row["nb_tx_hour"]) else row["nb_tx_hour"],
            axis=1
        )

    if "tx_amount_hour_mean" in df.columns and "tx_hour" in df.columns:
        fallback_mean = df.groupby("tx_hour")["tx_amount_hour_mean"].median()
        df["tx_amount_hour_mean"] = df.apply(
            lambda row: fallback_mean[row["tx_hour"]] if pd.isna(row["tx_amount_hour_mean"]) else row["tx_amount_hour_mean"],
            axis=1
        )
        
    # -------------------------
    # tx_amount_variation (evita infinitos por divisão por zero)
    # -------------------------
    if "tx_amount_variation" in df.columns:
        df["tx_amount_variation"] = df["tx_amount_variation"].replace([np.inf, -np.inf], np.nan)
        df["tx_amount_variation"] = df["tx_amount_variation"].fillna(0.0)
        
    return df