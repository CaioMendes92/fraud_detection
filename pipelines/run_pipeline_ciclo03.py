#1. Ler o arquivo raw (teste) -- ok
#2. Ler o arquivo parquet (histórico com features) -- ok
#3. Construir as features no teste (usando o histórico) -- ok (está em src)
#4. Tratar dados nulos -- ok (está em src)
#5. Selecionar as features usadas pelo modelo -- ok
#6. Rodar o modelo para gerar scores/predições 
#7. Retornar e salvar os resultados (score, label, id, etc.)
#8. (opcional) Logar/monitorar para rastreabilidade
def run_pipeline():
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    
    from feature_store.feature_metadata import FEATURE_METADATA, gerar_feature_book
    from src.feature_engineering import feature_engineering_new_data
    from src.treating_meassing_data import treating_meassing_data
    from src.load_data import load_data, merge_profiles
    from src.data_clean import data_clean
    import joblib
    import json
    from datetime import datetime

    # Timestamp para rastreio
    dt = datetime.now().strftime("%Y%m%d_%H%M")

    print("[1/9] Carregando dados...")
    test_df = load_data("data/raw/test_data.csv")
    train_df = load_data("data/processed/ciclo02/train_df_with_full_features_v2.parquet")

    print("[2/9] Enriquecendo dados com perfis de cliente e terminal...")
    test_df = merge_profiles(
        test_df,
        "data/external/new_customer_profiles_table.csv",
        "data/external/new_terminal_profiles_table.csv"
    )
    
    print("[3/9] Limpando dados...")
    test_df = data_clean(test_df)
    
    print("[4/9] Aplicando engenharia de features...")
    test_df = feature_engineering_new_data(train_df, test_df)
    
    print("[5/9] Tratando valores ausentes...")
    test_df = treating_meassing_data(test_df)

    print("[6/9] Selecionando features utilizadas pelo modelo...")
    book_df = gerar_feature_book(test_df, FEATURE_METADATA)
    feature_list = book_df.query("in_model == True")["nome"].tolist()

    missing = set(feature_list) - set(test_df.columns)
    if missing:
        raise ValueError(f"Faltam features no DataFrame: {missing}")
    
    X_test = test_df[feature_list]

    print("[7/9] Carregando modelo LightGBM e gerando predições...")
    model = joblib.load("models/lightgbm_model_ciclo03.pkl")
    test_df["score"] = model.predict_proba(X_test)[:, 1]
    test_df["fraude_predita"] = (test_df["score"] >= 0.3).astype(int)

    output_path = f"data/output/predicoes_v3_{dt}.parquet"
    print(f"[8/9] Salvando resultados em: {output_path}")
    test_df[["transaction_id", "score", "fraude_predita"]].to_parquet(output_path, index=False)

    metadata = {
        "versao_modelo": "v3",
        "data_execucao": dt,
        "features_utilizadas": feature_list,
        "arquivo_entrada": "test_data.csv",
        "arquivo_modelo": "lightgbm_model_ciclo03.pkl",
        "score_threshold": 0.3
    }
    
    with open(f"data/output/metadata_execucao_{dt}.json", "w") as f:
        json.dump(metadata, f, indent=2)
        
    print("[9/9] Pipeline finalizado com sucesso.")

if __name__ == '__main__':
    run_pipeline()