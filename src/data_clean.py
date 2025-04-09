import pandas as pd
def data_clean(df):
    if 'tx_datetime' in df.columns:
        df['tx_datetime'] = pd.to_datetime(df['tx_datetime']) 
    return df