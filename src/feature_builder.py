import pandas as pd
import numpy as np
import math
import json

class FeatureBuilder:
    def __init__(self, train_df, test_df, feature_book_path):
        self.train_df = train_df.copy()
        self.test_df = test_df.copy()

        with open(feature_book_path, "r", encoding="utf-8") as f:
            self.feature_book = json.load(f)

        self.features_por_grupo = self._organizar_por_grupo()
        self.reference_stats = self._compute_reference_stats()

    def _organizar_por_grupo(self):
        grupos = {}
        for feat, meta in self.feature_book.items():
            if meta["in_model"]:
                grupo = meta["grupo"]
                grupos.setdefault(grupo, []).append(feat)
        return grupos

    def _compute_reference_stats(self):
        df = self.train_df.copy()
        stats = {
            "mean_amount_per_customer": df.groupby("customer_id")["tx_amount"].mean(),
            "std_amount_per_customer": df.groupby("customer_id")["tx_amount"].std().replace(0, 1e-6),
            "median_amount_per_customer": df.groupby("customer_id")["tx_amount"].median(),
            "count_tx_per_hour": df.groupby(["customer_id", df["tx_datetime"].dt.hour])["tx_amount"].count(),
            "mean_tx_amount_per_hour": df.groupby(["customer_id", df["tx_datetime"].dt.hour])["tx_amount"].mean(),
        }
        df["time_since_last_tx"] = df.groupby("customer_id")["tx_datetime"].diff().dt.total_seconds()
        stats["mean_time_between_txs"] = df.groupby("customer_id")["time_since_last_tx"].mean()
        stats["std_time_between_txs"] = df.groupby("customer_id")["time_since_last_tx"].std().replace(0, 1e-6)
        return stats

    def _haversine(self, lon1, lat1, lon2, lat2):
        lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
        lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
        delta_lat = lat2_rad - lat1_rad
        delta_lon = lon2_rad - lon1_rad
        a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return 6371.0 * c

    def add_temporal_features(self):
        df = self.test_df
        df = df.sort_values(by=["customer_id", "tx_datetime"])
        df["time_since_last_tx"] = df.groupby("customer_id")["tx_datetime"].diff().dt.total_seconds()
        df["tx_hour"] = df["tx_datetime"].dt.hour
        df["weekday_of_day"] = df["tx_datetime"].dt.weekday
        df["month"] = df["tx_datetime"].dt.month
        df["year"] = df["tx_datetime"].dt.year
        df["is_weekend"] = df["tx_datetime"].dt.dayofweek >= 5
        df["month_reference"] = df["tx_datetime"].dt.strftime('%Y%m')
        df["tx_day"] = df["tx_datetime"].dt.day
        df["tx_month"] = df["tx_datetime"].dt.month
        df["tx_period_day"] = pd.cut(
            df["tx_hour"],
            bins=[0, 6, 12, 18, 24],
            labels=["madrugada", "manhã", "tarde", "noite"],
            right=False
        )
        self.test_df = df
        return self

    def add_espacial_features(self):
        df = self.test_df
        df["distance_customer_terminal"] = df.apply(
            lambda row: self._haversine(row["x_customer_id"], row["y_customer_id"],
                                        row["x_terminal_id"], row["y_terminal_id"]),
            axis=1
        )
        self.test_df = df
        return self

    def add_flags(self):
        df = self.test_df

        df["is_single_tx_customer"] = df.groupby("customer_id")["transaction_id"].transform("count") == 1

        df["repeated_tx"] = df.duplicated(subset=["customer_id", "tx_amount"], keep=False)

        df["consecutive_transactions_same_terminal"] = (
            df["terminal_id"] == df.groupby("customer_id")["terminal_id"].shift(1)
        )

        df["outlier_tx"] = np.abs(df["tx_amount"] - df["mean_amount"]) > (2 * df["std_amount"])

        df["high_value_tx"] = df["tx_amount"] > (3 * df["mean_amount"])

        df["frequent_tx"] = df["time_since_last_tx"] < 60

        df["unusual_hour"] = (df["tx_hour"] < 6) | (df["tx_hour"] > 22)

        self.test_df = df
        return self

    def add_variacao_features(self):
        df = self.test_df
        df["tx_amount_variation"] = df.groupby("customer_id")["tx_amount"].transform(lambda x: x.pct_change().fillna(0))
        self.test_df = df
        return self

    def add_estatisticas_referencia(self):
        df = self.test_df
        stats = self.reference_stats
        df["tx_hour"] = df["tx_datetime"].dt.hour
        df["mean_amount"] = df["customer_id"].map(stats["mean_amount_per_customer"])
        df["std_amount"] = df["customer_id"].map(stats["std_amount_per_customer"])
        df["median_amount"] = df["customer_id"].map(stats["median_amount_per_customer"])
        df["tx_amount_to_mean_ration"] = df["tx_amount"] / df["mean_amount"]
        df["tx_amount_median_ratio"] = df["tx_amount"] / df["median_amount"]
        df["outlier_tx"] = np.abs(df["tx_amount"] - df["mean_amount"]) > (2 * df["std_amount"])
        df["high_value_tx"] = df["tx_amount"] > (3 * df["mean_amount"])
        df["nb_tx_hour"] = df.set_index(["customer_id", "tx_hour"]).index.map(stats["count_tx_per_hour"])
        df["tx_amount_hour_mean"] = df.set_index(["customer_id", "tx_hour"]).index.map(stats["mean_tx_amount_per_hour"])
        df["mean_time_between_txs"] = df["customer_id"].map(stats["mean_time_between_txs"])
        df["std_time_between_txs"] = df["customer_id"].map(stats["std_time_between_txs"])
        self.test_df = df
        return self

    def add_historico_comportamental(self):
        from copy import deepcopy
        train_df = self.train_df.copy()
        test_df = self.test_df.copy()

        train_df["is_test"] = False
        test_df["is_test"] = True

        full_df = pd.concat([train_df, test_df], ignore_index=True).sort_values(["customer_id", "tx_datetime"])
        full_df["is_single_tx_customer"] = full_df.groupby("customer_id")["transaction_id"].transform("count") == 1
        full_df["repeated_tx"] = full_df.get("repeated_tx").fillna(
            full_df.duplicated(subset=["customer_id", "tx_amount"], keep=False)
        )
        full_df["consecutive_transactions_same_terminal"] = full_df.get("consecutive_transactions_same_terminal").fillna(
            full_df["terminal_id"] == full_df.groupby("customer_id")["terminal_id"].shift(1)
        )
        full_df["tx_hour"] = full_df["tx_datetime"].dt.hour
        full_df["outlier_tx"] = full_df["outlier_tx"].fillna(
            np.abs(full_df["tx_amount"] - full_df["mean_amount"]) > (2 * full_df["std_amount"])
        )
        full_df["high_value_tx"] = full_df["high_value_tx"].fillna(
            full_df["tx_amount"] > (3 * full_df["mean_amount"])
        )
        full_df["time_since_last_tx"] = full_df["time_since_last_tx"].fillna(
            full_df.groupby("customer_id")["tx_datetime"].diff().dt.total_seconds()
        )
        full_df["frequent_tx"] = full_df["frequent_tx"].fillna(full_df["time_since_last_tx"] < 60)
        full_df["unusual_hour"] = (full_df["tx_hour"] < 6) | (full_df["tx_hour"] > 22)

        self.test_df = full_df[full_df["is_test"]].drop(columns=["is_test"])
        return self

    def add_rolling_windows(self):
        window_sizes = ['1h', '2h', '4h', '8h', '12h', '24h']
        full_df = pd.concat([self.train_df.assign(is_test=False), self.test_df.assign(is_test=True)])
        full_df = full_df.sort_values(["customer_id", "tx_datetime"])

        results = []

        for cust_id, group in full_df.groupby("customer_id"):
            group = group.sort_values("tx_datetime").copy()
            group.set_index("tx_datetime", inplace=True)

            for window in window_sizes:
                rolled = (
                    group[["tx_amount"]]
                    .rolling(window=window, min_periods=1)
                    .agg({
                        "tx_amount": ["count", "sum", "mean", "std", "median", "max"]
                    })
                )
                rolled.columns = [f"{stat}_amount_per_customer_last_{window}" for stat in ["count", "total", "mean", "std", "median", "max"]]
                group = pd.concat([group, rolled], axis=1)

                std_col = f"std_amount_per_customer_last_{window}"
                mean_col = f"mean_amount_per_customer_last_{window}"
                z_col = f"amount_zscore_per_customer_last_{window}"

                group[std_col] = group[std_col].replace(0, 1e-6)
                group[z_col] = (group["tx_amount"] - group[mean_col]) / group[std_col]

            group.reset_index(inplace=True)
            results.append(group)

        full_df = pd.concat(results, ignore_index=True)

        # Adiciona ratios com base em 1h
        for window in window_sizes[1:]:
            full_df[f'ratio_mean_amount_per_customer_last_1h_to_{window}'] = (
                full_df['mean_amount_per_customer_last_1h'] / full_df[f'mean_amount_per_customer_last_{window}']
            )
            full_df[f'ratio_total_transactions_per_customer_last_1h_to_{window}'] = (
                full_df['count_amount_per_customer_last_1h'] / full_df[f'count_amount_per_customer_last_{window}']
            )

        self.test_df = full_df[full_df["is_test"]].drop(columns=["is_test"])
        return self


    def build(self):
        return self.test_df
