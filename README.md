# Non Supervised Machine Learning

Ficha Técnica: Proyecto de Análisis de Datos Predictivo

Título del Proyecto: Machine Learning No Supervisado

Objetivo:
Construir un modelo de machine learning.

Equipo:
Trabajo Individual.

Herramientas y Tecnologías:
- Python
- sklearn
- Google Colab

Procesamiento y análisis:
- limpieza de datos
- preprocesado de datos
- exploración de datos
- Técnica de Análisis de datos predictivo
  
Resultados y Conclusiones:
se realizaron modelos de machine learning(Logistic Regression, Linear Regression), se realizaron métricas para medir el rendimiento y regularizacion L1(Lasso).

Aggomerative Clustering:

```python
# aqui utilizo un modelo de machine learning llamado agglomerative clustering y lo entreno para hacer predicciones
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage='ward')
hc.fit(X)
y_hc = hc.fit_predict(X)
```
Diagrama de Codo:
```python
# aqui hago uso del modelo KMeans para hacer un diagrama de codo y determinar con este el numero adecuado de clusterers para mi anterior modelo de ML
from sklearn.cluster import KMeans
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(df_f)
    inertia.append(kmeans.inertia_)
#diagrama de codo
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Número de clústeres (k)')
plt.ylabel('Inercia')
plt.title('Diagrama de Codo')
plt.show()
```
Evaluación del modelo:
```python
# aqui hago un modelo de agglomerative clustering con los 4 clusters que me indica el diagrama de codo
model = AgglomerativeClustering(n_clusters=4)
# Evaluamos el modelo
cv = KFold(n_splits=10, random_state=1, shuffle=True)
silhouette_scores = []
for train_index, test_index in cv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    model.fit(X_train)
    labels = model.labels_
    score = silhouette_score(X_train, labels)
    silhouette_scores.append(score)
# Imprimimos el rendimiento
print('Silhouette Score: %.3f (%.3f)' % (np.mean(silhouette_scores), np.std(silhouette_scores)))
```

Limitaciones/Próximos Pasos:
Identifica y describe cualquier limitación o desafío encontrado durante el proyecto.
Sugiere posibles próximos pasos para extender o mejorar el proyecto de análisis de datos.


