import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_cardio = pd.read_csv("cardio_train.csv", sep=",", index_col=0)
df_cardio

df_cardio.columns

# 1. Análise geral dos dados
df_cardio.info()
df_cardio.describe()
df_cardio.isna().sum()

# 2. Análise exploratória dos dados (EDA)
df_cardio.columns




#2.1. Dados numéricos
from plotly.subplots import make_subplots
import plotly.graph_objects as go


fig = make_subplots(rows=4, cols=1)
fig.add_trace(go.Box(x=df_cardio["age"]/365, name="Idade"), row=1, col=1)
fig.add_trace(go.Box(x=df_cardio["weight"]/365, name="Peso"), row=2, col=1)
fig.add_trace(go.Box(x=df_cardio["ap_hi"]/365, name="Pressão sanguínea sistólica"), row=3, col=1)
fig.add_trace(go.Box(x=df_cardio["ap_lo"]/365, name="Pressão sanguínea diastólica"), row=4, col=1)


fig.update_layout(height=700)
fig.show()

# 2.2. Dados categóricos
df_cardio["gender"].value_counts()

from plotly.subplots import make_subplots

fig = make_subplots(rows=2, cols=3)

fig.add_trace(go.Bar(y=df_cardio["gender"].value_counts(), x=["Feminino", "Masculino"], name="Genero"), row=1,col=1)  


fig.update_layout(template="plotly_dark",height=700)
fig.show()

