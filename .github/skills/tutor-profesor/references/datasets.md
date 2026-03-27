# Guía de Datasets del Curso

Descripción detallada de los datasets utilizados en el curso, su estructura y usos pedagógicos.

## Datasets de Pandas (Clases 1-2)

### arbolado-en-espacios-verdes.csv

**Ubicación**: `clases/Clase-01-PythonPandas/practica/`

**Descripción**: Inventario de árboles en espacios verdes de la Ciudad Autónoma de Buenos Aires.

**Columnas principales**:
- `long`: Longitud geográfica
- `lat`: Latitud geográfica
- `id_especie`: Identificador de especie
- `nombre_cientifico`: Nombre científico del árbol
- `nombre_comun`: Nombre común
- `diametro_altura_pecho`: Diámetro del tronco (cm)
- `altura`: Altura del árbol (m)

**Tamaño**: ~10,000 filas

**Usos pedagógicos**:
- Operaciones básicas de Pandas: filtrado, agrupación, agregación
- Análisis exploratorio: especies más comunes, distribución de alturas
- Manejo de valores nulos
- Visualización: histogramas, boxplots

**Ejemplo de análisis**:
```python
# Top 5 especies más comunes
df['nombre_comun'].value_counts().head()

# Altura promedio por especie
df.groupby('nombre_comun')['altura'].mean().sort_values(ascending=False)
```

---

### arbolado-publico-lineal-2017-2018.csv

**Ubicación**: `clases/Clase-01-PythonPandas/practica/`

**Descripción**: Árboles en veredas y avenidas de CABA (arbolado lineal).

**Diferencia con el anterior**: Este dataset incluye árboles en vías públicas, no en parques.

**Usos pedagógicos**:
- Combinar datasets (merge/concat)
- Comparar distribuciones geográficas
- Detectar duplicados

---

### EncuestaDeMovilidad.csv

**Ubicación**: `clases/Clase-02-introMetodología/practica02/`

**Descripción**: Datos de encuesta sobre medios de transporte utilizados.

**Columnas principales**:
- `sexo`: Género del encuestado
- `edad`: Edad del encuestado
- `medio_transporte`: Tipo de transporte (auto, colectivo, tren, etc.)
- `tiempo_viaje`: Tiempo en minutos
- `distancia`: Distancia recorrida

**Usos pedagógicos**:
- Análisis de metodología de recolección de datos
- Identificación de sesgos en muestras
- Agrupaciones y agregaciones complejas
- Visualización de distribuciones

---

## Datasets de Calidad de Datos (Clase 10)

### DatosDengueYZikaCorregida.csv

**Ubicación**: `clases/Clase-10-CalidadDeDatos/practica10/`

**Descripción**: Datos epidemiológicos de casos de dengue y zika.

**Problemas de calidad intencionales** (para detectar):
- Valores faltantes en fechas
- Inconsistencias en formatos de fecha
- Outliers en número de casos
- Errores tipográficos en nombres de provincias

**Usos pedagógicos**:
- Detección de anomalías
- Limpieza de datos
- Imputación de valores faltantes
- Validación de rangos

---

## Datasets de Visualización (Clases 11-12)

### tips.csv

**Ubicación**: `clases/Clase-11-12-Visualizacion AED/practica11-12/`

**Descripción**: Propinas en un restaurante (dataset clásico de Seaborn).

**Columnas**:
- `total_bill`: Monto total de la cuenta
- `tip`: Propina
- `sex`: Género del cliente
- `smoker`: Si es fumador
- `day`: Día de la semana
- `time`: Almuerzo o cena
- `size`: Tamaño del grupo

**Usos pedagógicos**:
- Scatter plots: relación cuenta-propina
- Boxplots: propinas por día/género
- Heatmaps: correlaciones
- Facet grids: múltiples gráficos por categoría

**Ejemplo de análisis**:
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Relación cuenta vs propina
sns.scatterplot(data=df, x='total_bill', y='tip', hue='sex')
plt.title('Relación Cuenta-Propina por Género')
plt.show()

# Propinas por día
sns.boxplot(data=df, x='day', y='tip')
plt.title('Distribución de Propinas por Día')
plt.show()
```

---

### wine.csv

**Ubicación**: `clases/Clase-11-12-Visualizacion AED/practica11-12/`

**Descripción**: Características químicas de vinos y su calidad.

**Columnas** (13 features):
- `alcohol`, `malic_acid`, `ash`, `alcalinity_of_ash`, etc.
- `class`: Tipo de vino (1, 2 o 3)

**Usos pedagógicos**:
- Heatmaps de correlación
- PCA para visualización 2D
- Clustering
- Clasificación multiclase

---

### zoo.csv

**Ubicación**: `clases/Clase-11-12-Visualizacion AED/practica11-12/`

**Descripción**: Características de animales (pelo, plumas, huevos, etc.) y su clasificación.

**Usos pedagógicos**:
- Análisis de variables categóricas
- Agrupaciones por características binarias
- Árboles de decisión (ML)

---

## Datasets de Machine Learning (Clases 13-18)

### Titanic

**Ubicación**: `clases/Clase-13-IntroModelado/titanic/`

**Descripción**: Pasajeros del Titanic y si sobrevivieron.

**Archivos**:
- `train.csv`: Datos de entrenamiento
- `test.csv`: Datos de prueba (sin etiqueta `Survived`)

**Columnas principales**:
- `Survived`: Sobrevivió (0 o 1) — **variable objetivo**
- `Pclass`: Clase del pasaje (1, 2, 3)
- `Name`, `Sex`, `Age`
- `SibSp`: Número de hermanos/cónyuges a bordo
- `Parch`: Número de padres/hijos a bordo
- `Fare`: Tarifa pagada
- `Embarked`: Puerto de embarque (C, Q, S)

**Usos pedagógicos**:
- Clasificación binaria
- Feature engineering: extraer título de `Name`
- Manejo de valores nulos (Age, Cabin)
- One-hot encoding de variables categóricas
- Evaluación con confusion matrix, accuracy, precision, recall

**Ejemplo de preprocesamiento**:
```python
# Rellenar edad con mediana
df['Age'].fillna(df['Age'].median(), inplace=True)

# Convertir sexo a numérico
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# One-hot encoding de Embarked
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Seleccionar features
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = df['Survived']
```

---

### Iris

**Ubicación**: `clases/Clase-13-IntroModelado/iris/`

**Descripción**: Medidas de flores (dataset clásico de ML).

**Columnas**:
- `sepal_length`, `sepal_width`, `petal_length`, `petal_width`: Medidas en cm
- `species`: Especie (setosa, versicolor, virginica) — **variable objetivo**

**Usos pedagógicos**:
- Clasificación multiclase
- K-NN, árboles de decisión, Random Forest
- Visualización 2D con PCA
- Clustering (si se ignora la etiqueta)

**Ejemplo de clasificación**:
```python
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

X = df.drop('species', axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print(classification_report(y_test, y_pred))
```

---

### arboles.csv

**Ubicación**: `clases/Clase-14-15-Clasificacion/practica14-15/`

**Descripción**: Características de árboles para clasificar especie(es).

**Usos pedagógicos**:
- Árboles de decisión
- Random Forest
- Visualización de árboles con graphviz

---

## Datasets de Regresión (Clases 16-17)

### ventaCasas.csv

**Ubicación**: `clases/Clase-11-12-Visualizacion AED/practica11-12/`

**Descripción**: Precio de casas según características.

**Columnas típicas**:
- `superficie`: Metros cuadrados
- `habitaciones`: Número de habitaciones
- `precio`: Precio de venta — **variable objetivo**

**Usos pedagógicos**:
- Regresión lineal simple (precio vs superficie)
- Regresión múltiple
- Evaluación con R², MSE, RMSE
- Visualización de la recta de regresión

---

### datos_tiempo_reaccion/

**Ubicación**: `clases/Clase-13-IntroModelado/datos_tiempo_reaccion/`

**Descripción**: Experimentos de tiempo de reacción.

**Usos pedagógicos**:
- Regresión
- Análisis de series temporales (si aplica)
- Detección de outliers

---

## Datasets de Clustering (Clase 19)

### cheetah.csv y cheetahRegion.csv

**Ubicación**: `clases/Clase-11-12-Visualizacion AED/practica11-12/`

**Descripción**: Datos sobre guepardos (posiblemente ubicación, características).

**Usos pedagógicos**:
- K-Means
- DBSCAN (si hay datos espaciales)
- Visualización de clusters

---

## Datasets Adicionales

### gaseosas.csv

**Descripción**: Consumo o ventas de gaseosas.

**Usos**: Análisis de series temporales, tendencias.

---

### precioBiodiesel.csv

**Descripción**: Evolución del precio del biodiesel.

**Usos**: Gráficos de línea, análisis de tendencias.

---

### snow.csv

**Descripción**: Datos meteorológicos (nieve).

**Usos**: Análisis exploratorio, visualización.

---

### airport.csv

**Descripción**: Aeropuertos (posiblemente ubicación, tráfico).

**Usos**: Mapas, análisis geoespacial.

---

### votacionGeneral.csv

**Descripción**: Resultados de votaciones.

**Usos**: Agregaciones, visualización de resultados electorales.

---

## Generación de Datasets Sintéticos

Para ejercicios puntuales, se pueden generar datasets con:

```python
import numpy as np
import pandas as pd

# Dataset simple de ventas
np.random.seed(42)
datos = {
    'producto': np.random.choice(['A', 'B', 'C'], 100),
    'precio': np.random.uniform(10, 100, 100),
    'cantidad': np.random.randint(1, 20, 100)
}
df = pd.DataFrame(datos)

# Dataset para regresión lineal
X = np.linspace(0, 10, 100)
y = 2.5 * X + np.random.normal(0, 1, 100)  # y = 2.5x + ruido
df = pd.DataFrame({'x': X, 'y': y})

# Dataset para clustering
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=300, centers=3, random_state=42)
df = pd.DataFrame(X, columns=['feature1', 'feature2'])
```

---

## Recomendaciones para Ejercicios

1. **Empezar con datasets pequeños** (<1000 filas) para que los estudiantes vean resultados rápido
2. **Usar datasets del mundo real** (arbolado, encuestas) para motivar el análisis
3. **Incluir problemas de calidad** en datasets avanzados (nulos, outliers) para enseñar limpieza
4. **Combinar datasets** (ej: árboles de espacios verdes + arbolado lineal) para practicar merges
5. **Generar datasets sintéticos** cuando se necesite practicar un concepto específico sin distracciones

---

## Fuentes de Datos Adicionales

Si se quiere expandir el curso, considerar:

- **Buenos Aires Data**: [data.buenosaires.gob.ar](https://data.buenosaires.gob.ar)
- **Kaggle Datasets**: [kaggle.com/datasets](https://www.kaggle.com/datasets)
- **UCI ML Repository**: [archive.ics.uci.edu/ml](https://archive.ics.uci.edu/ml)
- **Seaborn datasets**: Incluidos en la librería (`sns.load_dataset()`)

---

## Resumen por Módulo

| Módulo | Datasets Clave | Objetivos |
|--------|----------------|-----------|
| 1 | arbolado, movilidad | Pandas básico |
| 2 | movilidad, SQL generados | SQL y normalización |
| 3 | dengue/zika, tips, wine | Calidad y visualización |
| 4 | Titanic, Iris | Clasificación y regresión |
| 5 | wine, cheetah | Clustering y PCA |
