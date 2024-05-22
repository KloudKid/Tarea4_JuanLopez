#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Configurar y entrenar el modelo K-means
kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
kmeans.fit(data_scaled)

# Evaluar el modelo
labels_kmeans = kmeans.labels_
silhouette_kmeans = silhouette_score(data_scaled, labels_kmeans)
calinski_harabasz_kmeans = calinski_harabasz_score(data_scaled, labels_kmeans)

print(f"Coeficiente de Silhouette (K-means): {silhouette_kmeans}")
print(f"Índice de Calinski-Harabasz (K-means): {calinski_harabasz_kmeans}")

# Visualizar los resultados
plt.scatter(data_scaled[:, 2], data_scaled[:, 3], c=labels_kmeans, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 2], kmeans.cluster_centers_[:, 3], marker='x', s=100, c='red')
plt.xlabel('Annual Income (scaled)')
plt.ylabel('Spending Score (scaled)')
plt.title('K-means Clustering')
plt.show()
# Centros de clústeres de K-means
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
print("Centros de los clústeres (K-means):")
print(cluster_centers)

# Agregar etiquetas a los datos originales
data['Cluster_Kmeans'] = labels_kmeans

# Resumen de cada clúster
summary_kmeans = data.groupby('Cluster_Kmeans').mean()

print("Resumen de clústeres K-means:")
print(summary_kmeans)


# In[ ]:




