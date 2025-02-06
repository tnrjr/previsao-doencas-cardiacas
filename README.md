# Análise de Dados Cardíacos e Machine Learning

Este projeto realiza uma análise exploratória de dados (EDA) e treinamento de um modelo de Machine Learning para prever a presença de doenças cardíacas com base em um conjunto de dados de pacientes.

## 📌 Objetivos

1. **Carregar e entender os dados** do conjunto `cardio_train.csv`.
2. **Realizar uma análise exploratória** dos dados numéricos e categóricos.
3. **Treinar um modelo de Machine Learning** utilizando um `RandomForestClassifier` para prever doenças cardíacas.
4. **Avaliar o modelo** e verificar a importância das features.

---

## 📂 Estrutura do Código

### 🔹 1. Importação das Bibliotecas

O código utiliza as bibliotecas:
- `numpy`, `pandas`: manipulação de dados;
- `matplotlib`, `plotly`: visualização;
- `sklearn`: treinamento e avaliação do modelo;
- `shap`: explicabilidade do modelo.

### 🔹 2. Carregamento e Visualização dos Dados

```python
df_cardio = pd.read_csv("cardio_train.csv", sep=",", index_col=0)
df_cardio.info()
df_cardio.describe()
df_cardio.isna().sum()
```

### 🔹 3. Análise Exploratória dos Dados (EDA)

Foram criados gráficos de boxplot para dados numéricos e histogramas para dados categóricos utilizando `plotly`.

```python
fig = make_subplots(rows=4, cols=1)
fig.add_trace(go.Box(x=df_cardio["age"]/365, name="Idade"), row=1, col=1)
fig.add_trace(go.Box(x=df_cardio["weight"]/365, name="Peso"), row=2, col=1)
fig.add_trace(go.Box(x=df_cardio["ap_hi"]/365, name="Pressão Sanguínea Sistólica"), row=3, col=1)
fig.add_trace(go.Box(x=df_cardio["ap_lo"]/365, name="Pressão Sanguínea Diastólica"), row=4, col=1)
fig.show()
```

### 🔹 4. Treinamento do Modelo de Machine Learning

O modelo **Random Forest** foi treinado com `sklearn`:

```python
from sklearn.ensemble import RandomForestClassifier
ml_model = RandomForestClassifier(n_estimators=20, n_jobs=4, max_depth=4)
ml_model.fit(x_train, y_train)
```

### 🔹 5. Avaliação do Modelo

O modelo foi avaliado com `classification_report` e `confusion_matrix`:

```python
from sklearn.metrics import classification_report, confusion_matrix
predictions = ml_model.predict(x_test)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
```

### 🔹 6. Importância das Features

A importância das variáveis foi analisada com `permutation_importance`:

```python
from sklearn.inspection import permutation_importance
result = permutation_importance(ml_model, x_test, y_test, n_repeats=10, n_jobs=2)
```

E visualizada com `matplotlib`:

```python
fig, ax = plt.subplots()
ax.boxplot(result.importances[sorted_idx].T, vert=False, labels=x_test.columns[sorted_idx])
ax.set_title("Permutation Importances (Test set)")
plt.show()
```

---

## Como Executar o Projeto

1. Instale os pacotes necessários:
   ```bash
   pip install numpy pandas matplotlib plotly scikit-learn shap
   ```

2. Execute o script Python para carregar e analisar os dados.

---


