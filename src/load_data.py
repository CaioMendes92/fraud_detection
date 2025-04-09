import pandas as pd
import os

def load_data(path: str):
    
    ext = os.path.splitext(path)[-1].lower()

    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Formato de arquivo não suportado: {ext}")
    
    return df

def merge_profiles(test_df, customer_profile_path, terminal_profile_path):
    customer_profiles = load_data(customer_profile_path)
    terminal_profiles = load_data(terminal_profile_path)

    df = pd.merge(test_df, customer_profiles, on='customer_id', how='left')
    df = pd.merge(df, terminal_profiles, on='terminal_id', how='left')

    return df