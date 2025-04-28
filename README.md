# Análise Preditiva de Doenças Cardíacas com Machine Learning

## Introdução
As doenças cardíacas representam uma das principais causas de mortalidade global. Antecipar o risco de desenvolvimento dessas doenças, por meio da análise de dados clínicos, é fundamental para o diagnóstico precoce e intervenções médicas eficazes. Este projeto aplicou Análise Exploratória de Dados (EDA) e Machine Learning para construir um modelo de previsão de doenças cardíacas baseado em características de pacientes.

## Objetivos

- Carregar e compreender os dados do arquivo `cardio_train.csv`.
- Realizar Análise Exploratória de Dados (EDA) para identificar padrões e anomalias.
- Executar o pré-processamento e divisão adequada dos dados.
- Treinar um modelo de Machine Learning usando `RandomForestClassifier`.
- Avaliar o desempenho do modelo por meio de métricas robustas.
- Interpretar a importância dos fatores de risco identificados pelo modelo.

## Metodologia

### 1. Coleta e Carregamento dos Dados
O conjunto de dados `cardio_train.csv` inclui variáveis como idade, gênero, peso, altura, pressão arterial, colesterol e hábitos de vida. Inicialmente, foram realizadas inspeções de consistência e análises descritivas.

### 2. Análise Exploratória de Dados (EDA)
Foram gerados boxplots, histogramas e análises de distribuição para identificar relações entre variáveis e detectar outliers. Observou-se, por exemplo, discrepâncias nos valores de pressão arterial e peso.

### 3. Pré-processamento dos Dados
As variáveis independentes foram separadas da variável alvo (`cardio`). Realizou-se normalização das features e a divisão dos dados em treino (67%) e teste (33%), assegurando balanceamento adequado.

### 4. Treinamento do Modelo
O modelo Random Forest foi treinado inicialmente com parâmetros padrão. Posteriormente, aplicou-se GridSearchCV para otimização dos hiperparâmetros como `n_estimators`, `max_depth`, `min_samples_split` e `min_samples_leaf`, buscando maximizar o F1-score.

### 5. Avaliação do Modelo
O modelo otimizado alcançou:

- **Acurácia**: 73,90%
- **Recall para doença cardíaca**: 68%
- **F1-score para doença cardíaca**: 74%

A matriz de confusão demonstrou uma boa capacidade de distinguir corretamente pacientes doentes e saudáveis, reduzindo erros de classificação relevantes.

### 6. Interpretação dos Fatores de Risco
Utilizou-se a ferramenta SHAP para interpretar a influência de cada feature na previsão. Idade, pressão arterial sistólica, colesterol e IMC destacaram-se como variáveis mais importantes.

## Conclusões

Este projeto demonstra que é possível construir um modelo de Machine Learning eficiente para prever doenças cardíacas utilizando dados clínicos básicos. Embora o modelo alcance bons resultados, há espaço para melhorias através de:

- Limpeza de outliers extremos.
- Engenharia de novas features.
- Testes com algoritmos mais avançados como XGBoost e LightGBM.

O modelo é particularmente útil para sistemas de triagem, apoio à decisão clínica e estudos de medicina preditiva.

---

**Nota:** Para reproduzir os resultados ou utilizar o modelo, recomenda-se salvar o modelo final treinado (`best_model.pkl`) e seguir os passos de carregamento e inferência descritos no projeto.
