#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset
data = pd.read_csv('D:/Universidad/11º Semestre/Analisis de Datos/Tarea 4/Mall_Customers.csv')

# Mostrar las primeras filas del dataset
print(data.head())

# Realizar un análisis descriptivo
print(data.describe())

# Visualizar relaciones entre variables
sns.pairplot(data)
plt.show()

# Identificar valores atípicos
sns.boxplot(data['Annual Income (k$)'])
plt.show()

sns.boxplot(data['Spending Score (1-100)'])
plt.show()
from sklearn.preprocessing import StandardScaler

# Convertir la columna 'Gender' en numérica
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

# Escalar los datos
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.drop(columns=['CustomerID']))

# Verificamos la escala
print(pd.DataFrame(data_scaled, columns=data.columns[1:]).head())
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Realizar el clustering jerárquico
Z = linkage(data_scaled, method='ward')

# Dibujar el dendograma
plt.figure(figsize=(10, 7))
dendrogram(Z, truncate_mode='level', p=5)
plt.title('Dendograma del clustering jerárquico')
plt.xlabel('Clientes')
plt.ylabel('Distancia')
plt.show()

# Obtener las etiquetas de los clusters
labels_hierarchical = fcluster(Z, t=5, criterion='maxclust')

# Evaluar el modelo
silhouette_hierarchical = silhouette_score(data_scaled, labels_hierarchical)
calinski_harabasz_hierarchical = calinski_harabasz_score(data_scaled, labels_hierarchical)

print(f"Coeficiente de Silhouette (Jerárquico): {silhouette_hierarchical}")
print(f"Índice de Calinski-Harabasz (Jerárquico): {calinski_harabasz_hierarchical}")

# Visualizar los resultados
plt.scatter(data_scaled[:, 2], data_scaled[:, 3], c=labels_hierarchical, cmap='viridis')
plt.xlabel('Annual Income (scaled)')
plt.ylabel('Spending Score (scaled)')
plt.title('Hierarchical Clustering')
plt.show()
# Agregar etiquetas a los datos originales
data['Cluster_Hierarchical'] = labels_hierarchical

# Resumen de cada clúster
summary_hierarchical = data.groupby('Cluster_Hierarchical').mean()

print("Resumen de clústeres Jerárquicos:")
print(summary_hierarchical)


# In[ ]:




