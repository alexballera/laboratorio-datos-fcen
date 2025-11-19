#%%
# Descripción:
"""
Grupo: Malfatti
Integrantes:
- Francisco Sandoval
- Jeremías Mannino
- Marcos Illescas

Contenido: Implementación para el Trabajo Práctico n°2 de Laboratorio de Datos
Incluyen los análisis necesarios para resolver las consignas planteadas en el enunciado del TP.
"""
#%%===========================================================================
# Importamos los datasets que vamos a utilizar en este programa
#=============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay



# %%
# Carga de datos y separación de los conjuntos
path_to_file = 'G:/Mi unidad/uba/exactas/lab-datos/clases/tp2-assets/'
data=pd.read_csv(path_to_file + "mnist_c_fog_tp.csv")

X = data.iloc[:, 1:-1]
y = data.iloc[:, -1]

# %%
# Funciones

# De visualizacion

def plot_digito(imagen_data, title=""):
    """Muestra la imagen a partir del df sin el índice ni label"""
    plt.imshow(imagen_data.values.reshape(28, 28), cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.savefig(f'Graficos/Figura_3_{title}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
def analisis_distr_digitos(y):
    """Genera gráfico de distribución de números"""
    counts = y.value_counts().sort_index()
    counts.plot(kind='bar', figsize=(10, 5))
    plt.title("Distribución de dígitos")
    plt.xlabel("Dígitos")
    plt.ylabel("Cantidad")
    plt.savefig('Graficos/Figura_1.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_digito_prom(data, digito, title=""):
    """Genera imagen promedio para el digito ingresado"""
    subset = data[data['labels'] == digito]
    pixeles = subset.iloc[:, 1:-1]
    prom_imagen = pixeles.mean(axis=0).values.reshape(28, 28)
    plt.imshow(prom_imagen, cmap='gray')
    plt.title(f"Imagen promedio del dígito {digito}")
    plt.axis('off')
    plt.savefig(f'Graficos/Figura_2_{title}.png', dpi=300, bbox_inches='tight')
    plt.show()


# Cálculo de promedios y desviaciones estándar por dígito
prom_imagenes = [data[data['labels'] == digito].iloc[:, 1:-1].mean(axis=0) for digito in range(10)]
std_imagenes = [data[data['labels'] == digito].iloc[:, 1:-1].std(axis=0) for digito in range(10)]

# Para eleccion de atributos para predecir cualquier número

def factor_diferencias_entre_digitos(X_sub, y_sub, clase):
    factores = {}

    # Promedio y desviación estándar de la clase específica
    mean1 = prom_imagenes[clase]
    std1 = std_imagenes[clase]

    # Promedio y desviación estándar del resto de las clases
    mean2 = pd.concat([prom_imagenes[i] for i in range(10) if i != clase], axis=1).mean(axis=1)
    std2 = pd.concat([std_imagenes[i] for i in range(10) if i != clase], axis=1).mean(axis=1)

    # Cálculo del factor para cada píxel
    for pix in X_sub.columns:
        factor = abs(mean1[pix] - mean2[pix]) / (std1[pix] + std2[pix])
        factores[pix] = factor

    return pd.Series(factores)

# Para eleccion de atributos para predecir ceros y unos

def factor_diferencias_entre_digitos_binario(X_sub, y_sub, clase1, clase2):
    factores = {}
    # Seleccionamos las filas de cada clase:
    X_c1 = X_sub[y_sub == clase1]
    X_c2 = X_sub[y_sub == clase2]
    for col in X_sub.columns:
        mean1 = X_c1[col].mean()
        mean2 = X_c2[col].mean()
        std1 = X_c1[col].std()
        std2 = X_c2[col].std()
        # Calculamos el factor
        factor = abs(mean1 - mean2) / (std1 + std2)
        factores[col] = factor
    return pd.Series(factores)

# De pre-procesamiento

def bordes(imag):
    # Convertir a matriz 28x28 sin reshape en cada iteración
    imag = np.array(imag, dtype=int).reshape(28, 28)

    # Crear una matriz de ceros para almacenar los bordes
    borde_matriz = np.zeros((28, 28), dtype=bool)

    # Calcular diferencias con desplazamientos (arriba, abajo, izq, der)
    difArr = imag[:-2, 1:-1] - imag[1:-1, 1:-1] > 20  # Arriba
    difAbj = imag[2:, 1:-1] - imag[1:-1, 1:-1] > 20   # Abajo
    difIzq = imag[1:-1, :-2] - imag[1:-1, 1:-1] > 20  # Izquierda
    difDer = imag[1:-1, 2:] - imag[1:-1, 1:-1] > 20   # Derecha

    # Unir todas las comparaciones usando np.logical_or.reduce()
    borde_matriz[1:-1, 1:-1] = np.logical_or.reduce((difArr, difAbj, difIzq, difDer))

    return borde_matriz



def bloquesEnFila(imag, fil):
    return np.sum(imag[fil, 1:] != imag[fil, :-1]) + 1

def bloquesEnColumna(imag, col):
    columna = imag[:, col]  # Extraer la columna completa
    return np.count_nonzero(columna[:-1] != columna[1:]) + 1

def listaDeBloques(imag):
    borde_img = bordes(imag)  # Calcular bordes una sola vez
    
    bloques_filas = [bloquesEnFila(borde_img, i) for i in range(28)]
    bloques_columnas = [bloquesEnColumna(borde_img, i) for i in range(28)]
    
    return bloques_filas + bloques_columnas  # Concatenar listas

# %%
# Análisis exploratorio (Ejercicio 1)

# Distribución de clases
analisis_distr_digitos(y)


# Ejemplo de imágenes
plot_digito(X.iloc[3],"Ejemplo A")
plot_digito(X.iloc[6],"Ejemplo B")  

# Visualización de imagenes promedio de cada dígito
for digit in range(10):
    plot_digito_prom(data, digit, str(digit))
    
    
# %%

# Grafico de color de varianza de píxeles
varianzas = X.var()
indices_ordenados = varianzas.sort_values(ascending=False).index.tolist()
top_10 = [int(i) for i in indices_ordenados[:int(0.1 * len(varianzas))]] # Primeros 10%
top_10_to_30 = [int(i) for i in indices_ordenados[int(0.1 * len(varianzas)):int(0.3 * len(varianzas))]] # Del 10% al 30%
resto = [int(i) for i in indices_ordenados[int(0.3 * len(varianzas)):]] # Resto
importancia = np.empty(len(varianzas), dtype=int)# Creamos un grupo de color para el mapa
importancia.fill(2) # Lleno con importancia 2, después los voy modificando
importancia[top_10] = 0 # Modifico importancia
importancia[top_10_to_30] = 1 # Modifico importancia
importancia_map= importancia.reshape(28, 28)
cmap = ['red', 'yellow', 'blue']
plt.figure(figsize=(6, 6))
plt.imshow(importancia_map, cmap=plt.cm.colors.ListedColormap(cmap), interpolation='nearest')
plt.title("Mapa de Importancia de cada píxel")
cbar = plt.colorbar(ticks=[0,1,2])
cbar.ax.set_yticklabels(['Baja', 'Media', 'Alta'])
plt.savefig('Graficos/Figura_4.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
    
# Mapa de importancia de píxeles

factoresxdigito=[]
for i in range(10):
    factoresxdigito.append(factor_diferencias_entre_digitos(X, y, i))

matriz_factores=[float(np.array([factoresxdigito[n].iloc[i] for n in range(10)]).mean()) for i in range(784)]

matriz_factores = np.array(matriz_factores).reshape(28, 28)
plt.figure(figsize=(6, 6))
ax = plt.gca()

heatmap = ax.imshow(matriz_factores, cmap="Reds", interpolation="nearest")
plt.xlabel("Columnas", fontsize=12)
plt.ylabel("Filas", fontsize=12)
# Barra de color
cbar = plt.colorbar(heatmap, fraction=0.046, pad=0.04)
cbar.set_label("Factor de diferenciación", fontsize=12)
plt.savefig('Graficos/Figura_5.png', dpi=300, bbox_inches='tight')
plt.show()

#%%
# Ejercicio 2: Clasificación binaria
# Filtrar datos
binary_data = data[data['labels'].isin([0, 1])]
X_bin = binary_data.iloc[:, 1:-1]
y_bin = binary_data.iloc[:, -1]

#%%
# Hago un gráfico de torta para ver la proporción de 0 y 1
datos_pie_chart = [y_bin[y_bin == 0].count(), y_bin[y_bin == 1].count()]
plt.pie(datos_pie_chart, labels=['Porcentaje de 0', 'Porcentaje de 1'], autopct='%1.1f%%')
plt.savefig('Graficos/Figura_6.png', dpi=300, bbox_inches='tight')
plt.show()

#%%
# Separación train-test
X_train, X_test, y_train, y_test = train_test_split(X_bin, y_bin, test_size=0.2, random_state=42)

#%%

# Factor para distinguir entre 0 y 1
mask01 = y.isin([0, 1])
X01 = X[mask01]
y01 = y[mask01]
factores01 = factor_diferencias_entre_digitos_binario(X01, y01, 0, 1)
pixeles_sort = factores01.sort_values(ascending=False).index.tolist()
factores_ordenados = factores01.sort_values(ascending=False)
top_10_factores = {
    columna: round(factor, 4)
    for columna, factor in factores_ordenados.head(10).items()
}
print(top_10_factores)

# Mapa
dibujo = np.empty(784, dtype=int)
dibujo.fill(1)
indice_columna = {col : i for i, col in enumerate(X01.columns)}
for col in pixeles_sort[0:10]:
  dibujo[indice_columna[col]] = 0
for col in pixeles_sort[10:]:
  dibujo[indice_columna[col]] = 1

dibujo_map = dibujo.reshape(28, 28)

lista_color = ['white', 'black']
plt.figure(figsize=(6, 6))
plt.imshow(dibujo_map, cmap=plt.cm.colors.ListedColormap(lista_color))
plt.xlabel("Columnas", fontsize=12)
plt.ylabel("Filas", fontsize=12)
plt.savefig('Graficos/Figura_7B.png', dpi=300, bbox_inches='tight')
plt.show()

# Mapa de calor en función de factor 01
matriz_factores = factores01.values.reshape(28, 28)
plt.figure(figsize=(6, 6))
ax = plt.gca()

heatmap = ax.imshow(matriz_factores, cmap="Reds", interpolation="nearest")
plt.xlabel("Columnas", fontsize=12)
plt.ylabel("Filas", fontsize=12)
# Barra de color
cbar = plt.colorbar(heatmap, fraction=0.046, pad=0.04)
cbar.set_label("Factor de diferenciación", fontsize=12)
plt.savefig('Graficos/Figura_7A.png', dpi=300, bbox_inches='tight')
plt.show()


#%%
# Experimentos variando los 3 atributos en KNN
top_10_pixels = pixeles_sort[:10]

resultados_pruebas = []

# Genero todas las combinaciones 3 píxeles y evalúo
for i in range(len(top_10_pixels)):
    for j in range(i + 1, len(top_10_pixels)):
        for k in range(j + 1, len(top_10_pixels)):
            seleccion = [top_10_pixels[i], top_10_pixels[j], top_10_pixels[k]]
            seleccion.sort()
            seleccion = [int(i) for i in seleccion]

            # Extraigo los subconjuntos de entrenamiento y test con solo esos atributos:
            X_train_sel = X_train.iloc[:, seleccion]
            X_test_sel = X_test.iloc[:, seleccion]

            # Modelo KNN con k=3
            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(X_train_sel, y_train)
            y_pred = knn.predict(X_test_sel)
            exactitud = accuracy_score(y_test, y_pred)

            resultados_pruebas.append((seleccion, exactitud))

mejor_combinacion = max(resultados_pruebas, key=lambda x: x[1])
print(f"Mejor combinación: {mejor_combinacion[0]} | Exactitud: {mejor_combinacion[1]:.2f}")

#%%
# Hago la matriz de confusión
mejor_combinacion_modelo = [int(i) for i in mejor_combinacion[0]]
X_train_sel = X_train.iloc[:, mejor_combinacion_modelo]
X_test_sel = X_test.iloc[:, mejor_combinacion_modelo]

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_sel, y_train)
y_pred = knn.predict(X_test_sel)
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.savefig('Graficos/Figura_8.png', dpi=300, bbox_inches='tight')
plt.show()


#%%
# Se varía k y la cantidad de atributos (píxeles)

# Rango de atributos y de k a evaluar
rango_atributos = range(3, 11)  # de 3 a 10 atributos
rango_k = range(3, 11)          # de k=3 a k=10

# Matriz para almacenar las exactitudes: filas = cantidad de atributos, columnas = k
resultados = np.zeros((len(rango_atributos), len(rango_k)))

for i, n_atributos in enumerate(rango_atributos):
    # Seleccionar los primeros n_atributos
    seleccion = pixeles_sort[:n_atributos]
    seleccion = [int(i) for i in seleccion]
    # Subconjuntos de train y test
    X_train_sel = X_train.iloc[:, seleccion]
    X_test_sel = X_test.iloc[:, seleccion]

    for j, k in enumerate(rango_k):
        # Modelo KNN con k vecinos
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_sel, y_train)
        y_pred = knn.predict(X_test_sel)
        exactitud = accuracy_score(y_test, y_pred)
        resultados[i, j] = exactitud

print(resultados)

#%%
# Gráfico de calor de la matriz con las exactitudes
plt.figure(figsize=(8, 6))
im = plt.imshow(resultados, cmap='YlGnBu', aspect='auto')
plt.colorbar(im, label='Exactitud')
plt.xticks(ticks=np.arange(len(rango_k)), labels=list(rango_k))
plt.yticks(ticks=np.arange(len(rango_atributos)), labels=list(rango_atributos))
plt.xlabel('Cantidad de vecinos (k)')
plt.ylabel('Cantidad de atributos (píxeles)')
#plt.title('Exactitud del modelo KNN en función de k y de la cantidad de atributos')
plt.savefig('Graficos/Figura_9.png', dpi=300, bbox_inches='tight')
plt.show()

#%%
# Ejercicio 3: Clasificación multiclase
# Pre-procesamiento y filtrado de atributos

# Separo en conjuntos de datos de desarrollo, validacion y held-out
X_dev, X_heldout, y_dev, y_heldout = train_test_split(X.values, y, test_size=0.1, random_state=42)
X_train,X_perf,y_train,y_perf = train_test_split(X_dev, y_dev, test_size=0.2, random_state=43)

# Diccionario para almacenar los 10 píxeles más importantes por clase
pixeles_importantes = {}

# Conjunto para evitar duplicados
top_50_pixeles = set()

# Calcular los factores para cada clase y obtener los 10 más importantes
for clase in range(10):
    factores = factor_diferencias_entre_digitos(X, y, clase)
    top_5= factores.nlargest(5).index.tolist()
    pixeles_importantes[clase] = top_5
    top_50_pixeles.update(top_5)

# Convertir a lista con orden específico
top_50_pixeles = list(map(int, top_50_pixeles))

imagen = np.zeros((28, 28), dtype=np.uint8)

for pixel in top_50_pixeles:
    fila = pixel // 28  # División entera para encontrar la fila
    columna = pixel % 28  # Módulo para encontrar la columna
    imagen[fila, columna] = 255  # Poner el píxel en blanco

# Mostrar la imagen de los píxeles elegidos
plt.figure(figsize=(5,5))
plt.imshow(imagen, cmap="gray")
plt.savefig('Graficos/Figura_10.png', dpi=300, bbox_inches='tight')
plt.show()

# Generar lista de atributos pre-procesados
listaDeAtribs = [listaDeBloques(X_train[i]) for i in range(len(X_train))]


listaDeAtribsPerf = [listaDeBloques(X_perf[i]) for i in range(len(X_perf))] 

listaDeAtribsDev = [listaDeBloques(X_dev[i]) for i in range(len(X_dev))]

listaDeAtribsHeld = [listaDeBloques(X_heldout[i]) for i in range(len(X_heldout))]

# Filtrado

X_train_filtrado = X_train[:, top_50_pixeles]
X_perf_filtrado = X_perf[:, top_50_pixeles]

# %%

# Búsqueda de hiperparámetros para el modelo con atributos filtrados

tree = DecisionTreeClassifier()

# Con profundidades del 1 al 10

for i in range(1,11):
    treeO = DecisionTreeClassifier(max_depth=i)
    treeO.fit(X_train_filtrado,y_train)
    y_pred_perf = treeO.predict(X_perf_filtrado)
    print(f"Exactitud del modelo original en el conjunto de validación: {accuracy_score(y_perf, y_pred_perf):.2f}")
    print(treeO)
    
# %%
    
# Con otros hiperparámetros

or_param_grid = {'max_depth': range(5, 11,5),"min_impurity_decrease": np.linspace(0, 0.00005, 2),"ccp_alpha": np.linspace(0,0.00001,2)}
grid_search_or = GridSearchCV(tree, or_param_grid, cv=10) 
grid_search_or.fit(X_train_filtrado, y_train)
final_tree = grid_search_or.best_estimator_
final_tree.fit(X_train_filtrado,y_train)

y_pred_perf = final_tree.predict(X_perf_filtrado)
print(f"Exactitud del modelo original en el conjunto de validación: {accuracy_score(y_perf, y_pred_perf):.2f}")
print(final_tree)

# %%
# Entrenar el arbol final con todo el conjunto de desarrollo y medir performance.

final_tree.fit(X_dev,y_dev)
y_pred_heldout = final_tree.predict(X_heldout)
print(f"Exactitud en el held-out: {accuracy_score(y_heldout, y_pred_heldout):.2f}")
print(final_tree)

# Matriz de confusión
ConfusionMatrixDisplay.from_predictions(y_heldout, y_pred_heldout)
plt.savefig('Graficos/Figura_11 B.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Búsqueda de hiperparámetros para el modelo con atributos pre-procesados

# Con profundidades del 1 al 10

for i in range(1,11):
    treeNuev = DecisionTreeClassifier(max_depth=i)
    treeNuev.fit(listaDeAtribs,y_train)
    y_pred_perf = treeNuev.predict(listaDeAtribsPerf)
    print(f"Exactitud del modelo nuevo en el conjunto de validación: {accuracy_score(y_perf, y_pred_perf):.2f}")
    print(treeNuev)

# %%
# Con otros hiperparámetros

param_grid = {'max_depth': range(5, 11,5),"min_impurity_decrease": np.linspace(0, 0.0001, 3),"ccp_alpha": np.linspace(0,0.0005,2)} 
grid_search_18 = GridSearchCV(tree, param_grid, cv=5) # Cambiar cv para cambiar el k-fold
grid_search_18.fit(listaDeAtribs, y_train)
best_tree_18 = grid_search_18.best_estimator_
best_tree_18.fit(listaDeAtribs,y_train)

y_pred_perf = best_tree_18.predict(listaDeAtribsPerf)
print(f"Exactitud del 18 en el conjunto de validación: {accuracy_score(y_perf, y_pred_perf):.2f}")
print(best_tree_18)

# %%
# Entrenar el arbol final con todo el conjunto de desarrollo y medir performance.

best_tree_18.fit(listaDeAtribsDev,y_dev)
y_pred_heldout = best_tree_18.predict(listaDeAtribsHeld)
print(f"Exactitud en el held-out: {accuracy_score(y_heldout, y_pred_heldout):.2f}")
print(best_tree_18)

# Matriz de confusión
ConfusionMatrixDisplay.from_predictions(y_heldout, y_pred_heldout)
plt.savefig('Graficos/Figura_11 A.png', dpi=300, bbox_inches='tight')
plt.show()