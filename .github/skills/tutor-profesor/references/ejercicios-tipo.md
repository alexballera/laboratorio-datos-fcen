# Ejercicios Tipo por Módulo

Este documento contiene ejercicios modelo para cada módulo del curso, útiles para generar prácticas similares.

## Módulo 1: Python y Pandas

### Ejercicio 1.1: Operaciones básicas con DataFrames

```python
import pandas as pd

# Dataset: ventas de productos
datos = {
    'producto': ['Laptop', 'Mouse', 'Teclado', 'Monitor', 'Laptop'],
    'precio': [50000, 1500, 3000, 25000, 48000],
    'cantidad': [2, 10, 5, 3, 1],
    'categoria': ['Computación', 'Accesorios', 'Accesorios', 'Computación', 'Computación']
}

df = pd.DataFrame(datos)

# Tareas:
# 1. Calcular el total de venta por fila (precio * cantidad)
# 2. Filtrar productos de categoría 'Computación'
# 3. Calcular el precio promedio por categoría
# 4. Encontrar el producto más caro
```

### Ejercicio 1.2: Lectura y limpieza de datos

```python
# Dataset: arbolado-en-espacios-verdes.csv
# Tareas:
# 1. Leer el CSV con Pandas
# 2. Ver las primeras 10 filas
# 3. Identificar columnas con valores nulos
# 4. Calcular estadísticas descriptivas de columnas numéricas
# 5. Filtrar árboles de una especie específica
```

## Módulo 2: SQL

### Ejercicio 2.1: Consultas básicas

```sql
-- Dataset: estudiantes (id, nombre, edad, carrera_id)
-- Tareas:
-- 1. Seleccionar todos los estudiantes mayores de 20 años
-- 2. Contar cuántos estudiantes hay por carrera
-- 3. Obtener el promedio de edad
-- 4. Listar estudiantes ordenados por nombre alfabéticamente
```

### Ejercicio 2.2: Joins

```sql
-- Dataset: estudiantes + carreras
-- Tareas:
-- 1. INNER JOIN: estudiantes con nombre de carrera
-- 2. LEFT JOIN: incluir estudiantes sin carrera asignada
-- 3. Contar estudiantes por carrera (incluyendo carreras sin estudiantes)
-- 4. Listar carreras con más de 5 estudiantes
```

### Ejercicio 2.3: Subconsultas

```sql
-- Tareas:
-- 1. Estudiantes con edad mayor al promedio
-- 2. Carreras con la mayor cantidad de estudiantes
-- 3. Estudiantes de la carrera con menor promedio de edad
```

## Módulo 3: Visualización

### Ejercicio 3.1: Análisis exploratorio con gráficos

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset: tips.csv (propinas en restaurante)
# Tareas:
# 1. Histograma de montos de propinas
# 2. Scatter plot: monto total vs propina
# 3. Boxplot: propinas por día de la semana
# 4. Heatmap: correlación entre variables numéricas
```

### Ejercicio 3.2: Gráficos de tendencias

```python
# Dataset: precioBiodiesel.csv
# Tareas:
# 1. Gráfico de línea: precio a lo largo del tiempo
# 2. Agregar marca de promedio como línea horizontal
# 3. Identificar y marcar el precio máximo y mínimo
# 4. Agregar títulos y etiquetas apropiadas
```

## Módulo 4: Machine Learning Supervisado

### Ejercicio 4.1: Clasificación con Titanic

```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Dataset: titanic
# Tareas:
# 1. Cargar dataset y hacer exploración inicial
# 2. Seleccionar features relevantes (Pclass, Sex, Age, Fare)
# 3. Manejar valores nulos en Age
# 4. Convertir variables categóricas a numéricas
# 5. Dividir en train/test (80/20)
# 6. Entrenar un árbol de decisión
# 7. Calcular accuracy en test
# 8. Mostrar classification report
```

### Ejercicio 4.2: K-NN con Iris

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Dataset: iris
# Tareas:
# 1. Cargar dataset
# 2. Estandarizar features con StandardScaler
# 3. Probar diferentes valores de K (1, 3, 5, 10, 20)
# 4. Comparar accuracy en test para cada K
# 5. Graficar accuracy vs K
# 6. Elegir el mejor K
```

### Ejercicio 4.3: Regresión Lineal

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Dataset: ventaCasas.csv
# Tareas:
# 1. Predecir precio de casa basado en superficie
# 2. Entrenar modelo de regresión lineal
# 3. Calcular R² y MSE
# 4. Graficar: puntos reales + línea de regresión
# 5. Predecir precio de una casa de 100 m²
```

## Módulo 5: Machine Learning No Supervisado

### Ejercicio 5.1: Clustering con K-Means

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Dataset: generado sintético o wine.csv
# Tareas:
# 1. Cargar y estandarizar datos
# 2. Aplicar K-Means con K=3
# 3. Visualizar clusters en 2D (usando PCA si es necesario)
# 4. Calcular silhouette score
# 5. Probar con diferentes K y elegir el óptimo (método del codo)
```

### Ejercicio 5.2: PCA

```python
from sklearn.decomposition import PCA

# Dataset: iris o wine
# Tareas:
# 1. Aplicar PCA para reducir a 2 componentes
# 2. Graficar datos en el nuevo espacio 2D
# 3. Calcular varianza explicada por cada componente
# 4. Determinar cuántos componentes explican 95% de varianza
```

## Ejercicios Integradores

### Integrador 1: Pipeline completo de análisis

```python
# Dataset: arbolado-en-espacios-verdes.csv
# Tareas:
# 1. Cargar y limpiar datos (nulos, duplicados)
# 2. EDA: estadísticas, distribuciones, gráficos
# 3. Consultas SQL sobre el DataFrame (con DuckDB)
# 4. Visualizaciones: top 10 especies, distribución por barrio
# 5. Informe final con conclusiones
```

### Integrador 2: Predicción y evaluación

```python
# Dataset: a elección (Titanic, Iris, etc.)
# Tareas:
# 1. Definir problema (clasificación o regresión)
# 2. Limpieza y preparación de datos
# 3. Exploración y visualización
# 4. Feature engineering
# 5. Entrenar 3 modelos diferentes
# 6. Comparar con métricas apropiadas
# 7. Seleccionar mejor modelo y justificar
# 8. Predecir con datos nuevos
```
