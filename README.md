# Detecção de Fraude

![alt text](https://github.com/CaioMendes92/fraud_detection/blob/main/imgs/fraud_logo.jpg)

# 1. Entendimento de Negócio

### **Objetivo**
Com o aumento das transações eletrônicas em pontos de venda físicos, o uso de cartões de crédito tornou-se essencial no cotidiano. No entanto, essa facilidade também trouxe um aumento nas fraudes, incluindo transações fraudulentas em terminais físicos comprometidos e ataques direcionados a esses locais.
Fraudes em terminais físicos prejudicam financeiramente as instituições e afetam a confiança dos consumidores, além de impactar negativamente a reputação das empresas envolvidas. Por isso, enfretamos uma demanda crescente por soluções automatizadas que possam examinar rapidamente grandes volumes de transações e identificar padrões de fraude em terminais físicos.

**Objetivo**: 
Neste Case, o objetivo é desenvolver um modelo de machine learning para detecção de fraudes em cartões de crédito, focado exclusivamente em transações realizadas em terminais físicos.

Como não há um modelo prévio de detecção de fraude, todas as fraudes são contabilizadas como perdas. Assim, o baseline inicial será definido como:

$\text{total perdido}$ = $\text{total fraud}$

sendo: 
- $\text{total fraud}$ = soma de todos os valores das transações consideradas fraude no periodo avaliado

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
    - Realizar **cross-validation**.
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

### Ciclo 02:
- Criar a RF, XGBoost e LightGBM com todas as features novas.
- Calcular AUC-ROC, Recall, Precision, F1-Score.
- Avaliar o total perdido
- Fazer um feature importance e encontrar as melhores variáveis

### Ciclo 03:
- Criar a XGBoost e LightGBM com as melhores features
- Fazer Cross-Validation e Tunagem dos hiper parâmetros para encontrar os melhores hiperparâmetros.
- Calcular AUC-ROC, Recall, Precision, F1-Score.
- Avaliar o total perdido
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

## 6. Performance dos Modelos de Machine Learning
![alt text](https://github.com/CaioMendes92/fraud_detection/blob/main/imgs/classificacao_ciclo03.png)

Os modelos estão com desempenhos bastante similares.

- AUC-ROC (0.723 e 0.722) indica uma boa separação entre transações legítimas e fraudulentas. Com melhorias na engenharia de features ou no balanceamento das classes, é possível obter resultados ainda melhores.
- O recall é relativamente baixo, ou seja, os modelos estão deixando passar mais de 65% das fraudes.
- A precision é bastante alta: os modelos quase não erram falsos positivos (ou seja, poucas transações legítimas estão sendo classificadas como fraude), acertando mais de 90%.
- O melhor F1-Score (~0.50) demonstra um equilíbrio razoável entre precision e recall. Neste caso, optamos por privilegiar a experiência do cliente — preferimos aceitar o risco de uma fraude passar despercebida a barrar uma transação legítima.

Dentre os modelos avaliados, o **LightGBM** presentou o melhor desempenho geral, com **AUC-ROC, Precision e F1-score superiores**. Entretanto, devido à proximidade dos resultados e aos desvios observados, pode-se considerar um empate técnico entre os dois modelos avaliados.

---

# 7. Tradução do Erro em Métricas de Negócio
A principal métrica de negócio considerada é o **prejuízo final**, causado por fraudes não detectadas (falsos negativos) e por falsos positivos (transações legítimas classificadas erroneamente como fraude).

![alt text](https://github.com/CaioMendes92/fraud_detection/blob/main/imgs/prejuizo_final.png)

Ao longo dos ciclos, observamos uma redução significativa dos custos com fraudes. Já no ciclo 01, utilizando apenas as variáveis do modelo base, obtivemos uma redução de aproximadamente 60% nas perdas, representando uma economia de quase R$ 200 mil. 
No ciclo 02, após a aplicação de engenharia de features, a redução foi ainda maior — ultrapassando R$ 300 mil. 
Por fim, no ciclo 03, com seleção de variáveis e ajuste dos melhores hiperparâmetros, alcançamos uma redução total de **77% no custo inicial**, passando de `R$ 411.671,78` para apenas `R$ 94.039,41`.

**Observação:**  
- No futuro, poderá ser interessante reavaliar os modelos com novas variáveis e outros parâmetros, a fim de determinar qual deles apresenta, de fato, a melhor performance. Atualmente, os resultados indicam um empate técnico.

# 8. Próximos Passos (Em progresso!)
- **Construir um book de variáveis no S3 da AWS**
- **Integração dos dados via AWS Athena**
- **Início do processo de deploy** do modelo selecionado.
---

### **Considerações Finais**
Este projeto segue a metodologia CRISP-DM, garantindo uma abordagem estruturada e iterativa para o desenvolvimento de modelos de detecção de fraudes. A avaliação contínua e a otimização do modelo são fundamentais para minimizar perdas financeiras e aumentar a eficácia da detecção.
