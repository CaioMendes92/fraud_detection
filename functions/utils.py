import os
import sys
import pandas as pd
from sqlalchemy import create_engine

def csv_para_banco(caminho_csv, nome_tabela):
    '''
    Recebe o caminho do arquivo .csv, o nome da tabela e salva diretamento no banco.
    '''
    df = pd.read_csv(caminho_csv)
    
    conexao_banco = "mysql+mysqlconnector://root:@localhost/bronze"
    engine = create_engine(conexao_banco)
    
    df.to_sql(nome_tabela, con=engine, if_exists='append', index=False)
    print("Dados inseridos com sucesso!")

def ler_csvs_em_pastas_e_salvar_no_banco(diretorio_principal):
    '''
    Lê os arquivos CSVs dentro da pasta de diretorio_principal e utiliza a função ``csv_para_banco`` para salvar diretamente no banco.
    '''
    todos_os_dados = []
    
    for root, dirs, files in os.walk(diretorio_principal):
        for file in files:
            if file.endswith('.csv'):
                caminho_completo = os.path.join(root, file)
                dados = pd.read_csv(caminho_completo)
                file_name = file.split(".csv")[0]
                todos_os_dados.append(dados)
                
                csv_para_banco(caminho_completo,file_name)
