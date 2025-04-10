# Documentação Técnica — Projeto de Detecção de Fraude

**Versão:** 1.0  
**Data:** 10/04/2025  
**Autor:** [Caio Vitor Castro Mendes]

---

## 1. Visão Geral
Com o aumento das transações eletrônicas em pontos de venda físicos, o uso de cartões de crédito tornou-se essencial no cotidiano. No entanto, essa facilidade também trouxe um aumento nas fraudes, incluindo transações fraudulentas em terminais físicos comprometidos e ataques direcionados a esses locais. Fraudes em terminais físicos prejudicam financeiramente as instituições e afetam a confiança dos consumidores, além de impactar negativamente a reputação das empresas envolvidas. Por isso, enfretamos uma demanda crescente por soluções automatizadas que possam examinar rapidamente grandes volumes de transações e identificar padrões de fraude em terminais físicos.

**Objetivo**: Neste Case, o objetivo é desenvolver um modelo de machine learning para detecção de fraudes em cartões de crédito, focado exclusivamente em transações realizadas em terminais físicos.

Como não há um modelo prévio de detecção de fraude, todas as fraudes são contabilizadas como perdas. Assim, o baseline inicial será definido como:

total perdido  = total fraud

sendo:

total fraud = soma de todos os valores das transações consideradas fraude no periodo avaliado
Nosso objetivo é construir um modelo onde:

total modelo < total fraude

ou seja, reduzir o prejuízo financeiro causado por transações fraudulentas.
---

## 2. Arquitetura do Pipeline

- Leitura de dados de treino (parquet)
- Leitura de dados brutos (CSV)
- Enriquecimento com perfis (clientes e terminais)
- Engenharia de features
- Tratamento de nulos
- Filtragem de variáveis via `feature_book`
- Aplicação de modelo LightGBM
- Geração de score e predição
- Salvamento de resultado e log de metadados

_Exemplo de comando:_
```bash
python -m pipelines.run_pipeline_ciclo03
```

---

## 3. Engenharia de Features

As features são divididas por grupo (conforme `feature_book_v1.json`):

- **temporal**: hora, dia, mês, tempo desde última transação
- **flag**: comportamento suspeito ou repetitivo
- **espacial**: distância entre cliente e terminal
- **estatística**: média, desvio, z-score, etc.
- **rolling_window**: janelas móveis (1h, 4h, 24h)
- **relacional_temporal**: razões entre janelas
- **variação**: variação percentual no valor

O dicionário completo está em: `feature_store/feature_book_v1.json`

---

## 4. Modelos Utilizados

- **Modelo principal:** LightGBM
- **Métricas avaliadas:**
  - Total perdido pelas fraudes
  - AUC-ROC
  - F1-Score
  - Precision / Recall
- **Threshold de classificação:** 0.3
- **Modelo salvo em:** `models/lightgbm_model_ciclo03.pkl`

---

### 5. Resultados e Métricas

#### Modelo: LightGBM - Ciclo 03

| Métrica     | Valor   |
|-------------|---------|
| AUC-ROC     | 0.723   |
| F1-Score    | 0.503   |
| Precision   | 0.927   |
| Recall      | 0.345   |
| Threshold   | 0.3     |

#### Impacto Financeiro

- Prejuízo estimado sem modelo: **R$ 411.671,78**
- Prejuízo estimado com modelo: **R$ 92.662,06**
- **Redução de perdas:** **77,5%**

Os dados foram obtidos a partir dos arquivos:

- `classificacao_modelo_ciclo03.json`
- `comparativo_prejuizo.json`

---

## 6. Estrutura de Pastas

```
fraud_detection_project/
├── data/
│   ├── raw/
│   ├── processed/
|   ├── output/
│   └── external/
├── feature_store/
├── models/
|   └── metrics/
├── notebooks/
├── reports/
|   ├── classificacao/
|   ├── docs/
|   ├── prejuizos/
├── pipelines/
├── src/
```

---

## 7. Execução do Pipeline

Requisitos:
```
pip install -r requirements.txt
```

Execução:
```
python -m pipelines.run_pipeline_ciclo03
```

Saída esperada:
- Arquivo `.parquet` com predições
- Arquivo `.json` com metadados da execução

---

## 8. Próximos Passos

- [ ] Modularizar `FeatureBuilder` com base nos grupos
- [ ] Implementar logs estruturados
- [ ] Criar testes unitários para garantir a robustes do projeto
- [ ] Subir o código para AWS
- [ ] Criar API REST com FastAPI
- [ ] Deploy com versionamento via MLflow ou DVC

---

## 9. Autoria

- Autor: Caio Vitor Castro Mendes
- GitHub: [[CaioMendes92](https://github.com/CaioMendes92/Caio-Portfolio)]
- Contato: [caio.mendes@hotmail.com / https://www.linkedin.com/in/caio-mendes-6654751ba/]
