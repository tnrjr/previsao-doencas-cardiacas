#!/usr/bin/env python
# coding: utf-8

# In[53]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# In[54]:


df_cardio = pd.read_csv("cardio_train.csv", sep=",", index_col=0)


# In[55]:


df_cardio


# In[56]:


df_cardio.columns


# # 1. Análise geral dos dados

# In[57]:


df_cardio.info()


# In[58]:


17664.000000 / 365


# In[59]:


df_cardio.describe()


# In[60]:


df_cardio.isna().sum()


# # 2. Análise exploratória dos dados (EDA)

# In[61]:


df_cardio.columns


# ### 2.1. Dados numéricos

# In[62]:


from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig = make_subplots(rows=4, cols=1)
fig.add_trace(go.Box(x=df_cardio["age"]/365, name="Idade"), row=1, col=1)
fig.add_trace(go.Box(x=df_cardio["weight"]/365, name="Peso"), row=2, col=1)
fig.add_trace(go.Box(x=df_cardio["ap_hi"]/365, name="Pressão sanguínea sistólica"), row=3, col=1)
fig.add_trace(go.Box(x=df_cardio["ap_lo"]/365, name="Pressão sanguínea diastólica"), row=4, col=1)


fig.update_layout(height=700)
fig.show()


# ### 2.2. Dados categóricos
# 

# In[63]:


df_cardio["gender"].value_counts()


# In[64]:


from plotly.subplots import make_subplots

fig = make_subplots(rows=2, cols=3)

fig.add_trace(go.Bar(y=df_cardio["gender"].value_counts(), x=["Feminino", "Masculino"], name="Genero"), row=1,col=1)  


fig.update_layout(template="plotly_dark",height=700)
fig.show()


# # 3. Machine Learning

# ### 3.1. Preparação do modelo

# In[65]:


y = df_cardio["cardio"]
x = df_cardio.loc[:, df_cardio.columns != 'cardio']


# In[66]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)



# In[67]:


x_train


# In[68]:


y_test


# ### 3.2. Treinamento do modelo

# In[69]:


from sklearn.ensemble import RandomForestClassifier

ml_model = RandomForestClassifier(n_estimators=20, n_jobs=4, max_depth=4,)
ml_model.fit(x_train, y_train)


# In[70]:


x_test.iloc[0].to_frame().transpose()


# In[71]:


ml_model.predict(x_test.iloc[0].to_frame().transpose())


# In[72]:


y_train.iloc[1]


# ### 3.3 Avaliação do modelo

# In[73]:


predictions = ml_model.predict(x_test)


# In[74]:


from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))


# ### 3.4. Feature importance

# In[75]:


from sklearn.inspection import permutation_importance

result = permutation_importance(ml_model, x_test, y_test, n_repeats=10, n_jobs=2) 
sorted_idx = result.importances_mean.argsort()


# In[76]:


fig, ax = plt.subplots()
ax.boxplot(result.importances[sorted_idx].T, vert=False, labels=x_test.columns[sorted_idx])
ax.set_title("Permutation Importances (Test set)")
fig.tight_layout()
plt.show()


# In[77]:


import shap
explainer = shap.TreeExplainer(ml_model)


# In[78]:


shap_values = explainer.shap_values(x)


# 
