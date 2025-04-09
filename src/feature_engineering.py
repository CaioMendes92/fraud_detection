from feature_store.feature_metadata import FEATURE_METADATA, gerar_feature_book

import numpy as np
import pandas as pd
import math

def feature_engineering(df):
    import time
    start_time = time.time()

    def time_features(df):
        # Ordenar os dados por cliente e data para cálculo de diferenças de tempo
        df = df.sort_values(by=["customer_id", "tx_datetime"])
    
        # Criar feature de tempo desde a última transação do cliente
        df["time_since_last_tx"] = df.groupby("customer_id")["tx_datetime"].diff().dt.total_seconds()
        
        df["tx_hour"] = df["tx_datetime"].dt.hour
        df["weekday_of_day"] = df["tx_datetime"].dt.weekday
        df['month'] = df['tx_datetime'].dt.month
        df['year'] = df['tx_datetime'].dt.year
        df['is_weekend'] = df['tx_datetime'].dt.dayofweek >= 5
        df['month_reference'] = df['tx_datetime'].dt.strftime('%Y%m')

        df["tx_day"] = df["tx_datetime"].dt.day
        df["tx_month"] = df["tx_datetime"].dt.month
        
        # categoria por horario do dia
        df["tx_period_day"] = pd.cut(
            df["tx_hour"],
            bins=[0, 6, 12, 18, 24],
            labels=["madrugada", "manhã", "tarde", "noite"],
            right=False,
        )
        
        return df

    def haversine(lon1,lat1,lon2,lat2):
        # convertes de graus para radianos
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        delta_lat = lat2_rad - lat1_rad
        delta_lon = lon2_rad - lon1_rad

        # formula de haversine
        a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

        raio_terra = 6371.0

        distancia = raio_terra * c
        return distancia

    def add_geodistance(df):
        df['distance_customer_terminal'] = df.apply(
            lambda row: haversine(row['x_customer_id'], row['y_customer_id'], row['x_terminal_id'], row['y_terminal_id']),
            axis=1
        )
        return df

    def add_behavioral_flags(df):
        """Cria flags binárias de comportamento atípico ou padrão."""
        df["is_single_tx_customer"] = df.groupby("customer_id")["transaction_id"].transform("count") == 1
        df["repeated_tx"] = df.duplicated(subset=["customer_id", "tx_amount"], keep=False)
        df["outlier_tx"] = np.abs(df["tx_amount"] - df["mean_amount"]) > (2 * df["std_amount"])
        df["high_value_tx"] = df["tx_amount"] > (3 * df["mean_amount"])
        df["unusual_hour"] = (df["tx_hour"] < 6) | (df["tx_hour"] > 22)
        df["frequent_tx"] = df["time_since_last_tx"] < 60  # em segundos?
        df['consecutive_transactions_same_terminal'] = df['terminal_id'] == df.groupby('customer_id')['terminal_id'].shift(1)
        return df

    def aggregate_features(df):
        df = df.sort_values(by=["customer_id", "tx_datetime"])
        window_sizes = ['1h', '2h', '4h', '8h', '12h', '24h']
        
        results = []
        for cust_id, group in df.groupby("customer_id"):
            group = group.sort_values("tx_datetime").copy()
            group.set_index("tx_datetime", inplace=True)
    
            for window in window_sizes:
                # Todas as métricas de uma vez
                rolled = (
                    group[["tx_amount"]]
                    .rolling(window=window, min_periods=1)
                    .agg({
                        "tx_amount": ["count", "sum", "mean", "std", "median", "max"]
                    })
                )
    
                # Corrige nome das colunas multiindex
                rolled.columns = [f"{stat}_amount_per_customer_last_{window}" for stat in ["count", "total", "mean", "std", "median", "max"]]
    
                group = pd.concat([group, rolled], axis=1)
    
                # z-score
                group[f'std_amount_per_customer_last_{window}'] = group[f'std_amount_per_customer_last_{window}'].replace(0, 1e-6)
                group[f'amount_zscore_per_customer_last_{window}'] = (
                    (group["tx_amount"] - group[f'mean_amount_per_customer_last_{window}']) / group[f'std_amount_per_customer_last_{window}']
                )
    
            group.reset_index(inplace=True)
            results.append(group)
    
        df = pd.concat(results, ignore_index=True)
    
        # Calcular os ratios entre janelas (fora do loop por cliente para evitar recalcular a cada grupo)
        for window in window_sizes[1:]:  # começa do segundo item (p/ comparar com 1h)
            df[f'ratio_mean_amount_per_customer_last_1h_to_{window}'] = (
                df['mean_amount_per_customer_last_1h'] / df[f'mean_amount_per_customer_last_{window}']
            )
            df[f'ratio_total_transactions_per_customer_last_1h_to_{window}'] = (
                df['count_amount_per_customer_last_1h'] / df[f'count_amount_per_customer_last_{window}']
            )

        return df
    
    def add_value_statistics(df):
        """Cria estatísticas por cliente e razões com o valor atual."""
        df["mean_amount"] = df.groupby("customer_id")["tx_amount"].transform("mean")
        df["std_amount"] = df.groupby("customer_id")["tx_amount"].transform("std")
        df["tx_amount_median_ratio"] = df["tx_amount"] / df.groupby("customer_id")["tx_amount"].transform("median")
        df["tx_amount_to_mean_ration"] = df["tx_amount"] / df["mean_amount"]
        return df

    def add_hourly_patterns(df):
        """Adiciona padrões por hora do dia para cada cliente."""
        df["nb_tx_hour"] = df.groupby(["customer_id", "tx_hour"])["tx_amount"].transform("count")
        df["tx_amount_hour_mean"] = df.groupby(["customer_id", "tx_hour"])["tx_amount"].transform("mean")
        return df

    def add_time_between_txs_stats(df):
        """Adiciona estatísticas do tempo entre transações (assume que time_since_last_tx já foi criado)."""
        df["mean_time_between_txs"] = df.groupby("customer_id")["time_since_last_tx"].transform("mean")
        df["std_time_between_txs"] = df.groupby("customer_id")["time_since_last_tx"].transform("std")
        return df

    def add_tx_variation(df):
        """Adiciona a variação percentual de valor das transações."""
        df["tx_amount_variation"] = df.groupby("customer_id")["tx_amount"].transform(lambda x: x.pct_change().fillna(0))
        return df

    df = add_geodistance(df)
    df = aggregate_features(df)
    df = time_features(df)
    df = add_value_statistics(df)
    df = add_hourly_patterns(df)
    df = add_time_between_txs_stats(df)
    df = add_tx_variation(df)
    df = add_behavioral_flags(df)

    end_time = time.time()
    print(f"Tempo de execução: {end_time - start_time:.2f} segundos")
    return df

def feature_engineering_new_data(train_df, test_df, feature_list=None):
    def time_features(train_df):
        # Ordenar os dados por cliente e data para cálculo de diferenças de tempo
        train_df = train_df.sort_values(by=["customer_id", "tx_datetime"])
    
        # Criar feature de tempo desde a última transação do cliente
        train_df["time_since_last_tx"] = train_df.groupby("customer_id")["tx_datetime"].diff().dt.total_seconds()
        
        train_df["tx_hour"] = train_df["tx_datetime"].dt.hour
        
        train_df["weekday_of_day"] = train_df["tx_datetime"].dt.weekday
        
        train_df['month'] = train_df['tx_datetime'].dt.month
            
        train_df['year'] = train_df['tx_datetime'].dt.year
        train_df['is_weekend'] = train_df['tx_datetime'].dt.dayofweek >= 5
        train_df['month_reference'] = train_df['tx_datetime'].dt.strftime('%Y%m')
    
        train_df["tx_day"] = train_df["tx_datetime"].dt.day
        train_df["tx_month"] = train_df["tx_datetime"].dt.month
        
        # categoria por horario do dia
        train_df["tx_period_day"] = pd.cut(
            train_df["tx_hour"],
            bins=[0, 6, 12, 18, 24],
            labels=["madrugada", "manhã", "tarde", "noite"],
            right=False,
        )
        
        return train_df
    
    def haversine(lon1,lat1,lon2,lat2):
        # convertes de graus para radianos
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
    
        delta_lat = lat2_rad - lat1_rad
        delta_lon = lon2_rad - lon1_rad
    
        # formula de haversine
        a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
        raio_terra = 6371.0
    
        distancia = raio_terra * c
        return distancia
    
    def add_geodistance(train_df):
        train_df['distance_customer_terminal'] = train_df.apply(
            lambda row: haversine(row['x_customer_id'], row['y_customer_id'], row['x_terminal_id'], row['y_terminal_id']),
            axis=1
        )
        return train_df
        
    def add_tx_variation(train_df):
        """Adiciona a variação percentual de valor das transações."""
        train_df["tx_amount_variation"] = train_df.groupby("customer_id")["tx_amount"].transform(lambda x: x.pct_change().fillna(0))
        return train_df

    def compute_reference_stats(train_df):
        reference_stats = {}
        reference_stats["mean_amount_per_customer"] = train_df.groupby("customer_id")["tx_amount"].mean()
        reference_stats["std_amount_per_customer"] = train_df.groupby("customer_id")["tx_amount"].std().replace(0, 1e-6)
        reference_stats["median_amount_per_customer"] = train_df.groupby("customer_id")["tx_amount"].median()
        train_df["tx_hour"] = train_df["tx_datetime"].dt.hour
        ref_hour_group = train_df.groupby(["customer_id", "tx_hour"])["tx_amount"]
        reference_stats["count_tx_per_hour"] = ref_hour_group.count()
        reference_stats["mean_tx_amount_per_hour"] = ref_hour_group.mean()
        train_df = train_df.sort_values(by=["customer_id", "tx_datetime"])
        train_df["time_since_last_tx"] = train_df.groupby("customer_id")["tx_datetime"].diff().dt.total_seconds()
        reference_stats["mean_time_between_txs"] = train_df.groupby("customer_id")["time_since_last_tx"].mean()
        reference_stats["std_time_between_txs"] = train_df.groupby("customer_id")["time_since_last_tx"].std().replace(0, 1e-6)
        return reference_stats

    def apply_reference_stats(train_df, reference_stats):
        train_df = train_df.copy()
    
        if "tx_hour" not in train_df.columns:
            train_df["tx_hour"] = train_df["tx_datetime"].dt.hour
    
        # Mapear estatísticas por cliente
        train_df["mean_amount"] = train_df["customer_id"].map(reference_stats["mean_amount_per_customer"])
        train_df["std_amount"] = train_df["customer_id"].map(reference_stats["std_amount_per_customer"])
        train_df["median_amount"] = train_df["customer_id"].map(reference_stats["median_amount_per_customer"])
    
        # Features derivadas
        train_df["tx_amount_to_mean_ration"] = train_df["tx_amount"] / train_df["mean_amount"]
        train_df["tx_amount_median_ratio"] = train_df["tx_amount"] / train_df["median_amount"]
        train_df["outlier_tx"] = np.abs(train_df["tx_amount"] - train_df["mean_amount"]) > (2 * train_df["std_amount"])
        train_df["high_value_tx"] = train_df["tx_amount"] > (3 * train_df["mean_amount"])
        train_df = train_df.drop(columns=['median_amount'],axis=1)
        # Padrões por hora
        train_df["nb_tx_hour"] = train_df.set_index(["customer_id", "tx_hour"]).index.map(reference_stats["count_tx_per_hour"])
        train_df["tx_amount_hour_mean"] = train_df.set_index(["customer_id", "tx_hour"]).index.map(reference_stats["mean_tx_amount_per_hour"])
    
        # Mapear tempos médios e std
        train_df["mean_time_between_txs"] = train_df["customer_id"].map(reference_stats["mean_time_between_txs"])
        train_df["std_time_between_txs"] = train_df["customer_id"].map(reference_stats["std_time_between_txs"])
    
        return train_df

    def apply_historical_features(train_df, test_df, feature_list=None):
        train_df = train_df.copy()
        test_df = test_df.copy()
    
        train_df["is_test"] = False
        test_df["is_test"] = True
    
        full_df = pd.concat([train_df, test_df], ignore_index=True)
        full_df = full_df.sort_values(by=["customer_id", "tx_datetime"])
    
        # ===============================================================
        # 1. FLAGS COMPORTAMENTAIS COM HISTÓRICO
        # ===============================================================
    
        
        full_df["is_single_tx_customer"] = full_df.groupby("customer_id")["transaction_id"].transform("count") == 1
        full_df["repeated_tx"] = full_df["repeated_tx"].fillna(
            full_df.duplicated(subset=["customer_id", "tx_amount"], keep=False)
        )
        full_df["consecutive_transactions_same_terminal"] = full_df["consecutive_transactions_same_terminal"].fillna(
            full_df["terminal_id"] == full_df.groupby("customer_id")["terminal_id"].shift(1)
        )
    
        # Verifica se colunas necessárias já existem
    
        full_df["tx_hour"] = full_df["tx_datetime"].dt.hour
        
        full_df["outlier_tx"] = full_df["outlier_tx"].fillna(
                np.abs(full_df["tx_amount"] - full_df["mean_amount"]) > (2 * full_df["std_amount"])
            )
        full_df["high_value_tx"] = full_df["high_value_tx"].fillna(
                full_df["tx_amount"] > (3 * full_df["mean_amount"])
            )
        
        # time_since_last_tx — só preenche onde estiver faltando
        full_df["time_since_last_tx"] = full_df["time_since_last_tx"].fillna(
            full_df.groupby("customer_id")["tx_datetime"].diff().dt.total_seconds()
        )
    
        full_df["frequent_tx"] = full_df["frequent_tx"].fillna(full_df["time_since_last_tx"] < 60)
        full_df["unusual_hour"] = (full_df["tx_hour"] < 6) | (full_df["tx_hour"] > 22)
    
        # ===============================================================
        # 2. ROLLING WINDOWS COM HISTÓRICO
        # ===============================================================
    
        window_sizes = ['1h', '2h', '4h', '8h', '12h', '24h']
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
                rolled.columns = [
                f"{stat}_amount_per_customer_last_{window}" 
                for stat in ["count", "total", "mean", "std", "median", "max"]
            ]
                for col in rolled.columns:
                    group[col] = rolled[col].values
    
                std_col = f'std_amount_per_customer_last_{window}'
                mean_col = f'mean_amount_per_customer_last_{window}'
                z_col = f'amount_zscore_per_customer_last_{window}'
    
                # Substitui std = 0 por valor mínimo para evitar divisão por zero
                std_values = group[std_col].replace(0, 1e-6).values
                mean_values = group[mean_col].values
    
                group[z_col] = (group["tx_amount"].values - mean_values) / std_values
    
            group.reset_index(inplace=True)
            results.append(group)
    
        full_df = pd.concat(results, ignore_index=True)
    
        # Ratios entre janelas
        for window in window_sizes[1:]:
            full_df[f'ratio_mean_amount_per_customer_last_1h_to_{window}'] = (
                full_df['mean_amount_per_customer_last_1h'] / full_df[f'mean_amount_per_customer_last_{window}']
            )
            full_df[f'ratio_total_transactions_per_customer_last_1h_to_{window}'] = (
                full_df['count_amount_per_customer_last_1h'] / full_df[f'count_amount_per_customer_last_{window}']
            )
    
        # Separar test_df com histórico aplicado
        test_df_final = full_df[full_df["is_test"]].drop(columns=["is_test"])
        return test_df_final

    reference_stats = compute_reference_stats(train_df)
    test_df = apply_reference_stats(test_df, reference_stats)
    test_df = apply_historical_features(train_df, test_df)
    test_df = time_features(test_df)
    test_df = add_geodistance(test_df)
    test_df = add_tx_variation(test_df)
    test_df['tx_datetime'] = pd.to_datetime(test_df['tx_datetime']) 

    return test_df