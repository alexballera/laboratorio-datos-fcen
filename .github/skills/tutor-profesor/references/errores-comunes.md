# Errores Comunes de Estudiantes

Esta referencia documenta los errores más frecuentes que cometen los estudiantes en cada módulo, con explicaciones y soluciones.

## Python Básico

### Error 1: Confundir asignación (=) con comparación (==)

**Incorrecto:**
```python
if edad = 18:  # SyntaxError
    print("Es mayor de edad")
```

**Correcto:**
```python
if edad == 18:
    print("Tiene 18 años")
```

**Explicación**: `=` asigna un valor, `==` compara dos valores.

---

### Error 2: Indentación inconsistente

**Incorrecto:**
```python
def calcular_promedio(lista):
promedio = sum(lista) / len(lista)  # IndentationError
    return promedio
```

**Correcto:**
```python
def calcular_promedio(lista):
    promedio = sum(lista) / len(lista)
    return promedio
```

**Explicación**: Python usa indentación (4 espacios recomendado) para delimitar bloques.

---

### Error 3: Modificar lista mientras se itera

**Incorrecto:**
```python
numeros = [1, 2, 3, 4, 5]
for num in numeros:
    if num % 2 == 0:
        numeros.remove(num)  # Comportamiento impredecible
```

**Correcto:**
```python
numeros = [1, 2, 3, 4, 5]
numeros = [num for num in numeros if num % 2 != 0]
```

**Explicación**: Modificar la lista que estás recorriendo puede saltarse elementos o generar errores.

---

## Pandas

### Error 1: No asignar el resultado de operaciones

**Incorrecto:**
```python
df.dropna()  # Esto no modifica df
print(df)  # Todavía tiene nulos
```

**Correcto:**
```python
df = df.dropna()  # Asignar de vuelta
# O usar inplace=True:
df.dropna(inplace=True)
```

**Explicación**: Muchas operaciones de Pandas devuelven una **copia** del DataFrame, no modifican el original.

---

### Error 2: Confundir loc e iloc

**Incorrecto:**
```python
df.loc[0]  # Si el índice no es numérico, esto puede fallar
df.iloc['nombre']  # iloc solo acepta enteros
```

**Correcto:**
```python
df.iloc[0]  # Primera fila por posición
df.loc[df.index[0]]  # Primera fila por etiqueta de índice
```

**Explicación**: `iloc` usa posiciones numéricas, `loc` usa etiquetas de índice.

---

### Error 3: Intentar sumar columnas de texto con .sum()

**Incorrecto:**
```python
df.groupby('ciudad').sum()  # Error si hay columnas de texto
```

**Correcto:**
```python
df.groupby('ciudad')['poblacion'].sum()  # Especificar columna numérica
# O seleccionar solo numéricas:
df.groupby('ciudad').select_dtypes(include='number').sum()
```

**Explicación**: `sum()` sin especificar columna intenta sumar todas, incluyendo strings.

---

### Error 4: Olvidar reset_index() después de groupby

**Incorrecto:**
```python
resultado = df.groupby('categoria')['precio'].mean()
# resultado es una Series, no un DataFrame con columna 'categoria'
print(resultado['categoria'])  # KeyError
```

**Correcto:**
```python
resultado = df.groupby('categoria')['precio'].mean().reset_index()
# Ahora 'categoria' es una columna, no el índice
print(resultado['categoria'])
```

**Explicación**: `groupby` convierte las columnas agrupadas en índice. `reset_index()` las vuelve columnas normales.

---

### Error 5: Usar apply() cuando no es necesario

**Ineficiente:**
```python
df['doble'] = df['valor'].apply(lambda x: x * 2)
```

**Eficiente:**
```python
df['doble'] = df['valor'] * 2  # Operación vectorizada
```

**Explicación**: Las operaciones vectorizadas son mucho más rápidas que `apply()`.

---

## SQL

### Error 1: Olvidar alias en columnas calculadas

**Incorrecto:**
```sql
SELECT nombre, precio * cantidad
FROM ventas;
-- La segunda columna se llama algo como "?column?" o "precio * cantidad"
```

**Correcto:**
```sql
SELECT nombre, precio * cantidad AS total
FROM ventas;
```

**Explicación**: Siempre asignar alias a columnas calculadas para que sean legibles.

---

### Error 2: WHERE con agregación en lugar de HAVING

**Incorrecto:**
```sql
SELECT categoria, COUNT(*) AS cantidad
FROM productos
WHERE COUNT(*) > 5  -- Error: no se puede usar función de agregación en WHERE
GROUP BY categoria;
```

**Correcto:**
```sql
SELECT categoria, COUNT(*) AS cantidad
FROM productos
GROUP BY categoria
HAVING COUNT(*) > 5;
```

**Explicación**: `WHERE` filtra **antes** de agrupar, `HAVING` filtra **después** de agrupar.

---

### Error 3: No especificar ON en JOIN

**Incorrecto:**
```sql
SELECT *
FROM estudiantes, carreras;  -- Esto es un CROSS JOIN (producto cartesiano)
```

**Correcto:**
```sql
SELECT *
FROM estudiantes
JOIN carreras ON estudiantes.carrera_id = carreras.id;
```

**Explicación**: Sin `ON`, SQL combina cada fila de la primera tabla con cada fila de la segunda (producto cartesiano).

---

### Error 4: Confundir IN con ANY/ALL

**Incorrecto:**
```sql
SELECT nombre
FROM productos
WHERE precio > ANY(SELECT precio FROM productos WHERE categoria = 'A');
-- Devuelve productos con precio mayor a AL MENOS UNO de categoría A
```

**Correcto (si se quiere mayor a todos):**
```sql
SELECT nombre
FROM productos
WHERE precio > ALL(SELECT precio FROM productos WHERE categoria = 'A');
-- O más claro:
WHERE precio > (SELECT MAX(precio) FROM productos WHERE categoria = 'A');
```

**Explicación**: `ANY` = "al menos uno", `ALL` = "todos".

---

## Visualización

### Error 1: Gráficos sin títulos ni etiquetas

**Incorrecto:**
```python
plt.plot(x, y)
plt.show()
```

**Correcto:**
```python
plt.plot(x, y)
plt.title('Evolución de Ventas')
plt.xlabel('Mes')
plt.ylabel('Ventas ($)')
plt.show()
```

**Explicación**: Un gráfico sin contexto es inútil. Siempre agregar título y etiquetas.

---

### Error 2: Usar gráfico de línea para variables categóricas

**Incorrecto:**
```python
plt.plot(df['categoria'], df['ventas'])  # Las categorías no son continuas
```

**Correcto:**
```python
plt.bar(df['categoria'], df['ventas'])  # Barras para categorías
```

**Explicación**: Líneas implican continuidad; usar barras para categorías discretas.

---

### Error 3: No ajustar límites de ejes

**Problema:**
```python
plt.scatter(x, y)
# El gráfico puede tener mucho espacio vacío o cortar puntos
```

**Solución:**
```python
plt.scatter(x, y)
plt.xlim(min(x) - 1, max(x) + 1)
plt.ylim(0, max(y) * 1.1)
```

**Explicación**: Ajustar límites para que los datos se vean claramente.

---

## Machine Learning

### Error 1: No dividir en train/test antes de explorar

**Incorrecto:**
```python
# Exploración y limpieza en todo el dataset
df = df.dropna()
# Luego dividir
X_train, X_test, y_train, y_test = train_test_split(X, y)
```

**Correcto:**
```python
# Primero dividir
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Luego hacer feature engineering SOLO en train
```

**Explicación**: Si explorás/limpiás antes de dividir, estás "espiando" el conjunto de test (data leakage).

---

### Error 2: No estandarizar features en KNN

**Incorrecto:**
```python
# Features en diferentes escalas (ej: edad 20-80, salario 20000-100000)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
```

**Correcto:**
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # ¡Usar el mismo scaler!

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
```

**Explicación**: KNN usa distancias; variables con mayor escala dominan el cálculo.

---

### Error 3: Evaluar solo con accuracy en datos desbalanceados

**Incorrecto:**
```python
# Dataset: 95% clase negativa, 5% clase positiva
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
# Modelo que predice siempre negativo tendría 95% accuracy
```

**Correcto:**
```python
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Mirar precision, recall, F1-score
```

**Explicación**: En datos desbalanceados, accuracy es engañoso. Usar métricas específicas por clase.

---

### Error 4: Overfitting por usar todo el dataset en cross-validation

**Incorrecto:**
```python
from sklearn.model_selection import cross_val_score
# Usar cross_val_score en todo el dataset sin separar test
scores = cross_val_score(modelo, X, y, cv=5)
```

**Correcto:**
```python
# Primero separar test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Cross-validation SOLO en train
scores = cross_val_score(modelo, X_train, y_train, cv=5)
# Evaluación final en test
modelo.fit(X_train, y_train)
test_score = modelo.score(X_test, y_test)
```

**Explicación**: Cross-validation es para seleccionar hiperparámetros en train; test debe usarse UNA VEZ al final.

---

### Error 5: No usar random_state

**Incorrecto:**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Cada ejecución da resultados diferentes
```

**Correcto:**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Resultados reproducibles
```

**Explicación**: `random_state` fija la semilla aleatoria para reproducibilidad.

---

## Calidad de Datos

### Error 1: Eliminar todas las filas con algún nulo sin analizar

**Incorrecto:**
```python
df = df.dropna()  # Puede eliminar el 90% de los datos
```

**Correcto:**
```python
# Analizar primero
print(df.isnull().sum())
# Decidir estrategia por columna:
df['edad'].fillna(df['edad'].median(), inplace=True)  # Imputar
df.drop(columns=['columna_irrelevante_con_nulos'], inplace=True)  # Eliminar columna
df.dropna(subset=['columna_critica'], inplace=True)  # Eliminar filas solo si falta esto
```

**Explicación**: `dropna()` sin argumentos es drástico; analizar qué columnas tienen nulos y por qué.

---

### Error 2: Imputar con media sin considerar distribuciones

**Incorrecto:**
```python
df['salario'].fillna(df['salario'].mean(), inplace=True)
# Si la distribución es sesgada, la media puede no ser representativa
```

**Correcto:**
```python
# Para distribuciones sesgadas, usar mediana
df['salario'].fillna(df['salario'].median(), inplace=True)
# O imputar por grupos
df['salario'].fillna(df.groupby('profesion')['salario'].transform('median'), inplace=True)
```

**Explicación**: La media es sensible a outliers; la mediana es más robusta.

---

## Normalización de Bases de Datos

### Error 1: No identificar todas las dependencias funcionales

**Problema**: No detectar que una columna depende de otra, dejando redundancia.

**Solución**: Listar explícitamente todas las dependencias funcionales antes de normalizar.

---

### Error 2: Normalizar de más (desnormalización justificada)

**Problema**: Llegar a 3FN o BCNF pero generar demasiadas tablas que hacen las consultas muy complejas.

**Solución**: En algunos casos prácticos, mantener 2FN puede ser suficiente si justificás por qué (ej: performance en consultas).

---

## Resumen de Buenas Prácticas

1. **Siempre leer la documentación** cuando uses una función nueva
2. **Verificar tipos de datos** con `df.dtypes` o `type(variable)`
3. **Imprimir resultados intermedios** para entender qué hace cada paso
4. **Usar nombres descriptivos** para variables y DataFrames
5. **Comentar el código** explicando el "por qué", no el "qué"
6. **Probar en datasets pequeños** antes de ejecutar en millones de filas
7. **Guardar versiones** del código antes de hacer cambios grandes
8. **Pedir ayuda** cuando algo no funciona después de varios intentos
