# Análise de Dados Cardíacos e Predição de Doenças Cardíacas com Machine Learning

##  Introdução
As doenças cardíacas estão entre as principais causas de morte no mundo. A capacidade de prever a presença dessas doenças com base em dados clínicos pode ser um fator determinante para diagnósticos precoces e tratamentos eficazes. Este projeto utiliza técnicas de Análise Exploratória de Dados (EDA) e Machine Learning para construir um modelo preditivo de doenças cardíacas baseado em um conjunto de dados de pacientes.

##  Objetivos

1. **Carregar e compreender os dados** contidos no arquivo `cardio_train.csv`.
2. **Realizar Análise Exploratória de Dados (EDA)** para entender as distribuições e relações entre variáveis.
3. **Pré-processar os dados** e preparar para o treinamento do modelo.
4. **Treinar um modelo de Machine Learning** utilizando o algoritmo `RandomForestClassifier` para prever a presença de doenças cardíacas.
5. **Avaliar o desempenho do modelo** utilizando métricas como `classification_report` e `confusion_matrix`.
6. **Analisar a importância das variáveis** para interpretação dos fatores de risco.

##  Metodologia

O projeto segue uma abordagem estruturada baseada nos seguintes passos:

### 1. Coleta e Carregamento dos Dados
O conjunto de dados `cardio_train.csv` contém informações de pacientes, incluindo fatores como idade, gênero, pressão arterial, colesterol e IMC. O dataset foi carregado e explorado inicialmente para verificação de inconsistências e estatísticas descritivas.

### 2. Análise Exploratória de Dados (EDA)
A EDA foi conduzida utilizando visualização gráfica e estatísticas descritivas para identificar tendências, outliers e relações entre variáveis. Foram gerados histogramas e boxplots para variáveis numéricas.

### 3. Pré-processamento dos Dados
Os dados foram normalizados e divididos em conjuntos de treino e teste. As variáveis independentes foram separadas da variável alvo para garantir um treinamento adequado do modelo.

### 4. Treinamento do Modelo
O modelo foi treinado utilizando Random Forest, um algoritmo robusto para classificação. Parâmetros como `n_estimators` e `max_depth` foram ajustados para otimizar o desempenho do modelo.

### 5. Avaliação do Modelo
O modelo foi avaliado com métricas de classificação como precisão, recall, F1-score e matriz de confusão, proporcionando uma análise detalhada da qualidade da previsão.

### 6. Importância das Features
A importância das variáveis foi avaliada para identificar os fatores mais relevantes na previsão da doença cardíaca. Foram utilizadas técnicas de importância de permutação para melhor compreensão do impacto de cada feature no modelo.

##  Métodos de Estimação
Para avaliar o desempenho do modelo, utilizou:
- **Matriz de confusão**: mostra os acertos e erros de classificação.
- **Precision, Recall e F1-score**: avaliam a qualidade da classificação para cada classe.
- **Importância das Features**: identifica os fatores mais relevantes na previsão da doença cardíaca.

Este projeto busca não apenas compreender a influência dos fatores de risco no desenvolvimento de doenças cardíacas, mas também destacar implicações clínicas e preventivas relevantes, especialmente em um contexto de avanços na medicina preditiva. Para detalhes metodológicos adicionais, consulte a seção correspondente.
