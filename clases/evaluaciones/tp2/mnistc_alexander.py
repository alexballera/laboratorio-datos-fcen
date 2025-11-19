# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 18:13:42 2025
Laboratorio de datos - Verano 2025
Trabajo Práctico 2:
    Clasificación y Selección de Modelos,
    utilizando validación cruzada
@author: Alexander Ballera
NOTA: Establecer este archivo como directorio
de trabajo, debido a que existen funciones en
el archivo modules.py los cuales importo acá
para ser reutilizados. Se realiza dando click
en el botón derecho del mouse sobre la pestaña
del archivo y dar click en 
"Establecer directorio de trabajo"
"""

#%% IMPORTS
import modules
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_validate

#%% ===========================================================================
# IMPORTO DATOS Y CREO DATAFRAME ORIGINAL
#%% ===========================================================================
path = 'G:/Mi unidad/uba/exactas/lab-datos/clases/tp2-assets/'

def get_data_all(path_to_file):
    data = pd.read_csv(path_to_file + "mnist_c_fog_tp.csv")
    return data

df = get_data_all(path)
#%% ===========================================================================
# 1.-ANÁLISIS EXPLORATORIO DE LOS DATOS
#%% ===========================================================================
#%%  Primeros análisis de los datos crudos
df.head()
#%% 
df.columns
#%% 
analisis_descriptivo_df = pd.DataFrame(modules.analisis_descriptivo(df))

#%% Observo columna labels
df['labels'].describe()

#%% # Elimino la columna 'Unnamed: 0'
def get_data(path_to_file):
    data = pd.read_csv(path_to_file + 'mnist_c_fog_tp.csv')
    data.pop('Unnamed: 0') 
    return data

data = get_data(path)
#%%  Preparo los datos, obtengo X, y
X, y = modules.get_data_x_y(data)

#%%  Análisis de los datos
analisis_descriptivo_df = pd.DataFrame(modules.analisis_descriptivo(X))

#%% Análisis descriptivo de cada dígito
def analisis_descriptivo_digitos():
    number = []
    rows = []
    mean = []
    sd = []
    cv = []
    min = []
    max = []
    nulls = []
    for i in range(0,10):
        number.append(str(i))
        rows.append(modules.analisis_descriptivo_digito(data, i)['rows'][0])
        mean.append(modules.analisis_descriptivo_digito(data, i)['mean'][0])
        sd.append(modules.analisis_descriptivo_digito(data, i)['sd'][0])
        cv.append(modules.analisis_descriptivo_digito(data, i)['cv'][0])
        min.append(modules.analisis_descriptivo_digito(data, i)['min'][0])
        max.append(modules.analisis_descriptivo_digito(data, i)['max'][0])
        nulls.append(modules.analisis_descriptivo_digito(data, i)['nulls'][0])
    return pd.DataFrame({'digit': number, 'rows': rows, 'min': min, 'max': max, 'mean': mean, 'sd': sd, 'cv': cv, 'nulls': nulls})

analisis_descriptivo_numeros = analisis_descriptivo_digitos()

#%% Análisis Descriptivo Números
def visualizacion_analisis_descriptivo_numeros():
    x = analisis_descriptivo_numeros['digit']
    counts = {
        'Mean': analisis_descriptivo_numeros['mean'],
        'SD': analisis_descriptivo_numeros['sd'],
    }
    width = 0.8
    
    fig, ax = plt.subplots()
    bottom = np.zeros(10)
    
    for label, count in counts.items():
        ax.bar(x, count, width, label=label, bottom=bottom)
        bottom += count
    
    ax.set_title('Análisis Descriptivo Dígitos')
    ax.set_xlabel('Dígitos')
    ax.set_ylabel('Cantidad')
    ax.legend()
    
    plt.show()

visualizacion_analisis_descriptivo_numeros()
#%% Visualización Cantidad de Filas x Dìgito
def visualizacion_cantidad_filas_digitos():
    fig, ax = plt.subplots()
    ax.bar(data = analisis_descriptivo_numeros, x='digit', height='rows')
    ax.set_title('Tuplas Por Dígitos')
    ax.set_xlabel('Dígitos')
    ax.set_ylabel('Cantidad')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 9000)
    ax.set_xticks(range(-1,10,1))
    ax.bar_label(ax.containers[0], fontsize=8)
    # plt.savefig('images/tuplas_por_digito.png', dpi=300, bbox_inches='tight')
    plt.show()

visualizacion_cantidad_filas_digitos()
#%%  Compruebo los primeros 10 dígitos (rows) para observar las imágenes resultantes

def visualizacion_diez_digitos():
    modules.plot_digito(X.iloc[1], plt, 'Label: 0', imagen_name='label_0')
    modules.plot_digito(X.iloc[3], plt, 'Label: 1', imagen_name='label_1')
    modules.plot_digito(X.iloc[5], plt, 'Label: 2', imagen_name='label_2')
    modules.plot_digito(X.iloc[7], plt, 'Label: 3', imagen_name='label_3')
    modules.plot_digito(X.iloc[2], plt, 'Label: 4', imagen_name='label_4')
    modules.plot_digito(X.iloc[0], plt, 'Label: 5', imagen_name='label_5')
    modules.plot_digito(X.iloc[13], plt, 'Label: 6', imagen_name='label6')
    modules.plot_digito(X.iloc[15], plt, 'Label: 7', imagen_name='label7')
    modules.plot_digito(X.iloc[17], plt, 'Label: 8', imagen_name='label8')
    modules.plot_digito(X.iloc[4], plt, 'Label: 9', imagen_name='label_9')

visualizacion_diez_digitos()
#%% Visualizo algunos dígitos de manera aleatoria

def visualizacion_digitos_aleatorios():
    ids = np.random.randint(0, X.shape[0], (10,1))
    fig, axs = plt.subplots(2,5)
    for i in range(2):
        for j in range(5):
            ind = ids[i*4 + j]
            axs[i,j].imshow(X.iloc[ind].values.reshape(28, 28), cmap='binary_r')
            axs[i,j].axis('off')
            axs[i,j].set_title(f'Label: {int(y.iloc[ind])}')
    plt.tight_layout()
    plt.show()

visualizacion_digitos_aleatorios()
#%% Visualizo los digitos por clases, usando su media
for i in range(0,10):
    modules.plot_digito(modules.get_data_digito(data, i)[0].mean(), plt, f'Imagen Promedio Label: {i}', imagen_name=f'imagen_promedio_label_{i}')

#%% Pruebo Dígitos con i = 0, 1, 2, 3, 4
for i in [0,1,5,7,8]:
    for j in range(1,6):
        modules.plot_digito(modules.get_data_digito(data, i)[0].iloc[j], plt, f"Dígito: {i}", imagen_name=f'imagen_prueba_label_{i}_{j}')
#%% Construccion df digitos

def construccion_df_digitos():
    d = []
    for i in range(10):
        d.append(modules.get_data_digito(data, i)[0])
    return d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9]

df_cero, df_uno, df_dos, df_tres, df_cuatro, df_cinco, df_seis, df_siete, df_ocho, df_nueve = construccion_df_digitos()
#%% ===========================================================================
# 2.-CLASIFICACIÓN BINARIA
#%% ===========================================================================
df_cero_uno_desc = analisis_descriptivo_numeros.iloc[0:2]
#%% Visualización Ceros vs Uno
def visualizacion_ceros_unos():
    fig, ax = plt.subplots()
    ax.bar(data = df_cero_uno_desc, x='digit', height='rows')
    ax.set_title('Cero vs Uno')
    ax.set_xlabel('Dígitos')
    ax.set_ylabel('Cantidad')
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 8500)
    ax.set_xticks(range(-1,2,1))
    ax.bar_label(ax.containers[0], fontsize=8)
    # plt.savefig('images/cero_vs_uno.png', dpi=300, bbox_inches='tight')
    plt.show()

visualizacion_ceros_unos()
#%% Visualización Ceros vs Uno Pie
def visualizacion_ceros_unos_pie():
    body_plot_pie = {
                    'datos': [int(df_cero_uno_desc['rows'][0]), int(df_cero_uno_desc['rows'][1])],
                    'labels': ['Cero', 'Uno'],
                    'colors': ['cornflowerblue', 'darksalmon'],
                    'image': 'images/cero_uno_pie.png',
                    'title': 'Cero vs Uno Subsets'
                    }
    
    modules.plot_pie(
        plt,
        body_plot_pie['datos'],
        body_plot_pie['labels'],
        body_plot_pie['colors'],
        body_plot_pie['image'],
        body_plot_pie['title']
        )

visualizacion_ceros_unos_pie()
#%% a.- Contruyo nuevo dataframe con 0 y 1
df_cero_uno = data[(data['labels'] == 0) | (data['labels'] == 1)]
X, y = modules.get_data_x_y(df_cero_uno)

#%% 
for i in range(0,20):
    modules.plot_digito(X.iloc[i], plt, 'Dígito')

#%% b.- Separo los datos en conjuntos de train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
#%% Descripción subsets
descripcion_subsets = {
    'labels': ['X_train', 'X_test', 'y_train', 'y_test'],
    'rows': [X_train.shape[0], X_test.shape[0], y_train.shape[0], y_test.shape[0]],
    'percent': [modules.percent(X_train.shape[0], len(X)), modules.percent(X_test.shape[0], len(X)), modules.percent(y_train.shape[0], len(X)), modules.percent(y_test.shape[0], len(X))]
    }
descripcion_subsets = pd.DataFrame(descripcion_subsets)
desc_subsets = pd.DataFrame(modules.analisis_descriptivo_subsets(X_train, X_test))
#%% Visualizacion subsets X
body_plot_pie = {
                'datos': [X_train.shape[0], X_test.shape[0]],
                'labels': ['Train', 'Test'],
                'colors': ['mediumseagreen', 'lightcoral'],
                'image': 'images/x_subsets_pie.png',
                'title': 'X Subsets Train & Test'
                }

modules.plot_pie(
    plt,
    body_plot_pie['datos'],
    body_plot_pie['labels'],
    body_plot_pie['colors'],
    body_plot_pie['image'],
    body_plot_pie['title']
    )

#%% Visualizacion subsets Y
body_plot_pie = {
                'datos': [y_train.shape[0], y_test.shape[0]],
                'labels': ['Train', 'Test'],
                'colors': ['slateblue', 'coral'],
                'image': 'images/y_subsets_pie.png',
                'title': 'Y Subsets Train & Test'
                }

modules.plot_pie(
    plt,
    body_plot_pie['datos'],
    body_plot_pie['labels'],
    body_plot_pie['colors'],
    body_plot_pie['image'],
    body_plot_pie['title']
    )

#%% Modelo KNeighborsClassifier
k = 3
model, y_pred = modules.model_knn(KNeighborsClassifier, k, X_test, X_train, y_train)

#%% Metrics
metricas, cm = modules.get_df_metrics(pd, metrics, model, np, X_train, X_test, y_train, y_test, y_pred)

#%% Visualizacion Mètricas
modules.visualizacion_metricas(plt, metricas, title='Métricas del Modelo KNN con k = 3')

#%% Visualizacion matriz de confusión con sns
modules.visualizacion_matriz_confusion_sns(plt, sns, cm)

#%% Visualizacion matriz de confusión con ConfusionMatrixDisplay
modules.visualizacion_matriz_confusion_sklearn(metrics, plt, y_test, y_pred)

#==============================================================================

#%% c.- Contruyo nuevo dataframe con 3 atributos random
attr = 10 # evalúo con 3 y luego con 10
attrs_df = modules.get_df_with_attr(df_cero_uno, np, attr)

#%% Construyo DF X e Y
X, y = modules.get_data_x_y(attrs_df)

#%% Separo los datos en conjuntos de train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

#%% Modelo KNeighborsClassifier
k = 3
model, y_pred = modules.model_knn(KNeighborsClassifier, k, X_test, X_train, y_train)

#%% Metrics
metricas, cm = modules.get_df_metrics(pd, metrics, model, np, X_train, X_test, y_train, y_test, y_pred)

#%% Visualizacion Mètricas
modules.visualizacion_metricas(plt, metricas)

#%% Visualizacion matriz de confusión con ConfusionMatrixDisplay
modules.visualizacion_matriz_confusion_sklearn(metrics, plt, y_test, y_pred)

#==============================================================================

#%% d.-Analizar modelo de KNN, diferentes atributos y diferentes k
data_all = data.copy()
rango_attr = range(3, 11)  # de 3 a 10 atributos
rango_k = range(3, 11)     # de 3 a 10 k

resultados = np.zeros((len(rango_attr), len(rango_k)))
# una fila por cada atributo, una columna por cada k

for i, attr in enumerate(rango_attr):
    
    for j, k in enumerate(rango_k):
        
        attrs_df = modules.get_df_with_attr(df_cero_uno, np, attr) # data_all df_cero_uno
        X, y = modules.get_data_x_y(attrs_df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
        model, y_pred = modules.model_knn(KNeighborsClassifier, k, X_test, X_train, y_train)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        
        print('i:', i, 'j:', j, 'attr:', attr, 'k:', k, 'accuracy:', accuracy)
        
        resultados[i, j] = accuracy

#%% Create an annotated heatmap
resultados = []
def visualizacion_resultados_heatmap(data=resultados,
                                     vmin = 0.0,
                                     vmax = 0.7,
                                     ylabel="Pixeles",
                                     cmap = 'RdPu',
                                     title = 'Exactitud del modelo KNN',
                                     xlabel = 'Cantidad Vecinos (K)',
                                     xticklabels=list(rango_attr),
                                     yticklabels=list(rango_k),
                                     ):
    plt.figure(figsize = (10,8))
    plt.rcParams.update({'font.size': 10})
    sns.heatmap(data,
                cmap = cmap, # 'RdPu' 'PiYG'
                vmin = vmin,
                vmax = vmax,
                center = 0,
                annot=True,
                fmt=".2f",
                square=True,
                linewidths=.5,
                xticklabels=xticklabels,
                yticklabels=yticklabels
                )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

visualizacion_resultados_heatmap(resultados)

#%% ===========================================================================
# 3.-CLASIFICACIÓN MULTICLASE
#%% ===========================================================================
# MODELO DecisionTreeClassifier 
#------------------------------------------------------------------------------
# Obtengo X, y de los datos
data_all = data.copy()
X, y = modules.get_data_x_y(data_all)

#%% experimento 1.-Separo valores dev y eval y creo modelos para kfolds = 5,10,15
# DEV

data_all = data.copy()
k = 5
dev_kf5, test_kf5 = modules.validacion_kfold(k,
                                        train_test_split,
                                        np,
                                        KFold, 
                                        tree,
                                        metrics,
                                        data_all
                                        )

k = 10
dev_kf10, test_kf10 = modules.validacion_kfold(k,
                                        train_test_split,
                                        np,
                                        KFold, 
                                        tree,
                                        metrics,
                                        data_all
                                        )

k = 15
dev_kf15, test_kf15 = modules.validacion_kfold(k,
                                        train_test_split,
                                        np,
                                        KFold, 
                                        tree,
                                        metrics,
                                        data_all
                                        )
#%%
alturas = list(range(1,11)) # modelos o cols
alturas = range(1,11)
k = 5 # 5, 10, 15

modelos = [
    {'k': 5, 'cmap': 'PiYG', 'data': dev_kf5, 'tipo': 'DEV'},
    {'k': 10, 'cmap': 'PiYG', 'data': dev_kf10, 'tipo': 'DEV'},
    {'k': 15, 'cmap': 'PiYG', 'data': dev_kf15, 'tipo': 'DEV'},
    {'k': 5, 'cmap': 'RdPu', 'data': test_kf5, 'tipo': 'TEST'},
    {'k': 10, 'cmap': 'RdPu', 'data': test_kf10, 'tipo': 'TEST'},
    {'k': 15, 'cmap': 'RdPu', 'data': test_kf15, 'tipo': 'TEST'}
    ]

for i in modelos:
    visualizacion_resultados_heatmap(
        i['data'],
        title=f'Exactitud del modelo {i['tipo']} {i['k']} K',  # 5, 10, 15
        xlabel = 'Altura hmax',
        ylabel = 'KFolds',
        vmin = 0.0,
        vmax = 0.7,
        cmap = i['cmap'], # PiYG
        xticklabels=list(alturas),
        yticklabels=list(range(i['k'])))

#%%
def scores_folds(data):
    mean = data.mean()
    sd = data.std()
    max = data.max()
    return mean, sd, max
#%%
dev_kf5_mean, dev_kf5_sd, dev_kf5_max = scores_folds(dev_kf5)
dev_kf10_mean, dev_kf10_sd, dev_kf10_max = scores_folds(dev_kf10)
dev_kf15_mean, dev_kf15_sd, dev_kf15_max = scores_folds(dev_kf15)

dev_mean = [dev_kf5_mean, dev_kf10_mean, dev_kf15_mean]
dev_sd = [dev_kf5_sd, dev_kf10_sd, dev_kf15_sd]
dev_max = [dev_kf5_max, dev_kf10_max, dev_kf15_max]
kfold = [5,10,15]
dev_scores = pd.DataFrame({'kfold': kfold, 'max': dev_max, 'mean': dev_mean, 'sd': dev_sd})

#%%
test_kf5_mean, test_kf5_sd, test_kf5_max = scores_folds(test_kf5)
test_kf10_mean, test_kf10_sd, test_kf10_max = scores_folds(test_kf10)
test_kf15_mean, test_kf15_sd, test_kf15_max = scores_folds(test_kf15)

test_mean = [test_kf5_mean, test_kf10_mean, test_kf15_mean]
test_sd = [test_kf5_sd, test_kf10_sd, test_kf15_sd]
test_max = [test_kf5_max, test_kf10_max, test_kf15_max]
kfold = [5,10,15]
test_scores = pd.DataFrame({'kfold': kfold, 'max': test_max, 'mean': test_mean, 'sd': test_sd})

#%% Análisis Descriptivo Números
def visualizacion_analisis_descriptivo_dev():
    x = dev_scores['kfold']
    counts = {
        'Mean': dev_scores['mean'],
        'SD': dev_scores['sd'],
    }
    width = 0.8
    
    fig, ax = plt.subplots()
    bottom = np.zeros(3)
    
    for label, count in counts.items():
        ax.bar(x, count, width, label=label, bottom=bottom)
        bottom += count
    
    ax.set_title('Análisis Descriptivo KFolds')
    ax.set_xlabel('Accuracy')
    ax.set_ylabel('KFolds')
    ax.legend()
    
    plt.show()

visualizacion_analisis_descriptivo_dev()


#%% entreno el modelo elegido en el conjunto dev entero
# arbol_elegido = tree.DecisionTreeClassifier(max_depth = 10)
# arbol_elegido.fit(X_dev, y_dev)
# y_pred = arbol_elegido.predict(X_dev)

# score_arbol_elegido_dev = metrics.accuracy_score(y_dev, y_pred)
# print(score_arbol_elegido_dev)

# #%% pruebo el modelo elegid y entrenado en el conjunto eval
# y_pred_eval = arbol_elegido.predict(X_eval)       
# score_arbol_elegido_eval = metrics.accuracy_score(y_eval, y_pred_eval)
# print(score_arbol_elegido_eval)

# #%% 
# # Matriz de confusion
# cm = metrics.confusion_matrix(y_eval, y_pred_eval)
# modules.visualizacion_matriz_confusion_sklearn(metrics, plt, y_eval, y_pred_eval)
# modules.visualizacion_matriz_confusion_sns(plt, sns, cm)

#%% ===========================================================================
# MODELO RandomForestClassifier 
#==============================================================================

data_all = data.copy()
X, y = modules.get_data_x_y(data_all)

# 1. Crear e inicializar una instancia del modelo
modelo = RandomForestClassifier(n_estimators=40)

# 2. Crear e inicializar una instancia de KFold
k_fold = KFold(n_splits=10, shuffle=True, random_state=42)

# 3. Realizar la validación cruzada usando "cross_validate"
scores = cross_validate(modelo, X, y, cv=k_fold, return_train_score=True)

scores
#%%
MEAN_DEV = [] # Promedio desempeños entrenamiento
STD_DEV = [] # Desviación desempeños entrenamiento
MEAN_VAL = [] # Promedio desempeños validación
STD_VAL = [] # Desviación desempeños validación
K = [] # Valores de k

# Realizar validación cruzada con k = 5, 15, 25, ... 95
for k in range(5,21,5):
    print(f'Validación cruzada con k = {k}')
    modelo = RandomForestClassifier(n_estimators=40)
    k_fold = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = cross_validate(modelo, X, y, cv=k_fold, return_train_score=True)

    # Calcular y almacenar desempeños (medias y desviaciones)
    # así como el valor de "k"
    MEAN_DEV.append(scores['train_score'].mean())
    STD_DEV.append(scores['train_score'].std())
    MEAN_VAL.append(scores['test_score'].mean())
    STD_VAL.append(scores['test_score'].std())
    K.append(k)

# Graficar desempeños entrenamiento y validación (media y desviación)
# vs. k
plt.errorbar(K, MEAN_DEV, yerr=STD_DEV, fmt='o', capsize=5, label='Entrenamiento')
plt.errorbar(K, MEAN_VAL, yerr=STD_VAL, fmt='o', capsize=5, label='Validación')
plt.xlabel('k')
plt.ylabel('Promedio ± desviación')

# Agregar leyenda
plt.legend()
#%%
