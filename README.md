# Detecção de Fraude

![alt text](https://github.com/CaioMendes92/fraud_detection/blob/main/imgs/fraud_logo.jpg)

# 1. Entendimento de Negócio

### **Objetivo**
O problema a ser resolvido envolve a identificação de transações fraudulentas. Atualmente, todas as transações são consideradas legítimas, resultando em perdas financeiras significativas para a empresa. Nosso objetivo é construir um modelo escalável, econômico e robusto para detectar essas fraudes.

Como não há um modelo prévio de detecção de fraude, todas as fraudes são contabilizadas como perdas. Assim, o baseline inicial será definido como:


$$\text{total perdido} = \text{total fraude}$$


onde:

- **total fraude** = soma de todos os valores de transações classificadas como fraudulentas.


Nosso objetivo é construir um modelo onde:

$$\text{total modelo} < \text{total fraude}$$

ou seja, reduzir o prejuízo financeiro causado por transações fraudulentas.

---

# 2. Entendimento dos Dados
Os dados utilizados foram obtidos a partir do livro:
- https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/SimulatedDataset.html

O conjunto de dados é composto por duas tabelas principais:
### 1. transactions
- **transaction_id**: Identificador único da transação.
- **tx_datetime**: Date e hora da transação.
- **customer_id**: Identificador único do cliente.
- **tx_amount**: Valor da transação.
- **tx_time_seconds**: Tempo da transação em segundos.
- **tx_fraud**: Variável binária indicando se a transação é fraudulenta (1) ou legítima (0).

### 2. Customer
- **customer_id**: Identificador único do cliente.
- **(x_customer_id,y_customer_id)**: Coordenadas geográficas do cliente em uma grade 100x100.
- **mean_amount / std_amount**: Média e desvio padrão dos valores das transações do cliente.

---

# 3. Preparação dos Dados

### **Objetivo**
Criar features robustas e eliminar informações desnecessárias para melhorar a performance do modelo.

### **Ciclos de Desenvolvimento**
- **Ciclo 01:**
    - Manter as variáveis originais.
- **Ciclo 02:**
    - Engenharia de features para criar novas variáveis.
    - Testar a exclusão e adição de novas features para treinar novamente o modelo.
- **Ciclo 03:**
    - Realizar **cross-validation** considerando a variável `datetime` para separar treino e teste.
    - Ajustar hiperparâmetros em conjunto com a **cross-validation**.
    - Selecionar o melhor modelo para produção.

---

# 4. Modelagem

### **Objetivo** 
Construir modelos preditivos e avaliar performance.
    
### Ciclo 01:
- Criar Random Forest, XGBoost e LightGBM com todas as features para obter um novo valor de baseline, baseado nos custos das transações.
- Calcular AUC-ROC, Recall, Precision, F1-Score.
- Analisar Feature Importance para identificar possíveis redundâncias.

---

# 5. Avaliação
### **Objetivo**
Garantir que o modelo atenda às necessidades da detecção de fraudes.

- Construir um novo total perdido baseado na matriz de confusão.

### **Métricas principais**
- **Matriz de confusão**
- **AUC-ROC**
- **Recall**
- **Precision**
- **F1-score**
A avaliação será feita comparando as perdas do modelo com a matriz de confusão.

---

## 6. Performance dos Modelos de Machine Learning (Em progresso!)
![alt text](https://github.com/CaioMendes92/fraud_detection/blob/main/imgs/modelo_ml_ciclo02.png)

Os resultados demonstram que o **XGBoost** apresentou o melhor desempenho geral, obtendo **AUC-ROC, Precision e F1-score superiores**.  

A **AUC-ROC mais alta** no conjunto de testes indica uma boa separação entre classes fraudulentas e legítimas.

Dessa forma, o XGBoost foi escolhido como referência para a construção da matriz de confusão:

![alt text](https://github.com/CaioMendes92/fraud_detection/blob/main/imgs/matriz_confusao_ciclo02.png)

**Análise da matriz de confusão**:
- **Baixa taxa de falsos positivos (0,17%)**: O modelo raramente classifica uma transação legítima como fraude, reduzindo o impacto para clientes legítimos.
- **Alta taxa de falsos negativos (78,70%)**: O modelo não detecta a maioria das fraudes, identificando corretamente apenas **21,30% das transações fraudulentas**.

---

# 7. Tradução do Erro em Métricas de Negócio (Em progresso!)
A métrica principal será o **prejuízo final** causado por fraudes não detectadas (falsos negativos) e pelos falsos positivos (transações legítimas classificadas erroneamente como fraude).

![alt text](https://github.com/CaioMendes92/fraud_detection/blob/main/imgs/classificacao_ciclo02.png)

Houve uma redução no custo geral, mas a melhoria entre o **Ciclo 1 e Ciclo 2** foi pequena. No entanto, no **Ciclo 2**, realizamos engenharia de features, diminuindo a dependência do modelo em relação a algumas variáveis, tornando-o mais robusto.

**Observação:**  
- O **XGBoost** teve métricas de performance melhores.  
- O **LightGBM** apresentou **melhor resultado financeiro**, possivelmente porque funciona melhor para transações de valores mais altos, enquanto o XGBoost tem melhor desempenho para valores menores.

# 8. Próximos Passos (Em progresso!)
- **Realizar uma cross-validation** com **Random Forest, XGBoost e LightGBM** para definir o modelo ideal.
- **Ajuste fino de hiperparâmetros** para otimizar os resultados.
- **Iniciar o processo de deploy** do modelo escolhido.

---

### **Considerações Finais**
Este projeto utiliza a metodologia **CRISP-DM** para garantir uma abordagem estruturada na modelagem de detecção de fraudes. A avaliação contínua e a otimização do modelo são essenciais para reduzir as perdas financeiras e aprimorar a eficácia da detecção.
