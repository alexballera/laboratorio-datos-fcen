# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 18:13:42 2025
Laboratorio de datos - Verano 2025
Trabajo Práctico 2:
    Clasificación y Selección de Modelos,
    utilizando validación cruzada
@author: Alexander Ballera
"""

#%% IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn import tree

#%% ===========================================================================
# IMPORTO DATOS Y CREO DATAFRAME ORIGINAL
#%% ===========================================================================
path_to_file = 'G:/Mi unidad/uba/exactas/lab-datos/clases/tp2-assets/'
df = pd.read_csv(path_to_file + 'mnist_c_fog_tp.csv')
df.pop('Unnamed: 0') # Elimino la columna 'Unnamed: 0', posteriormente lo explico

#%% ===========================================================================
# 1.-ANÁLISIS EXPLORATORIO DE LOS DATOS
#%% ===========================================================================
#%%  Primeros análisis de los datos crudos

df.head()
df.shape
df.columns
'''
Observo que la columna 'Unnamed: 0' es una variable incremental en 1 igual al
index, supondría que era el index de los datos originales. Además, observo
que la última columna es 'labels', la cual es diferente a las otras columnas
que van desde 0 hasta 783
'''
df.info()
df.mean()
df.std()

#%%  Verifico si existe algún valor nulo o campo vacío
def isna_validations(dataframe):
    return dataframe.isna().all().hasnans

isna_validations(df)

#%%  Preparo los datos
# df.pop('Unnamed: 0') # Elimino la columna 'Unnamed: 0' previamenle lo quité
X = df.copy() # Copio el df a la variable X para mantener el df original
X['labels'].describe()
y = X.pop('labels') # Escojo la columna 'labels' como la variable 'y'

#%%  Análisis de los datos
X.shape
X.info()
X.mean()
X.std()

#%%  Compruebo los primeros 10 dígitos (rows) para observar las imágenes resultantes

def test_digito(i):
    plt.imshow(X.iloc[i].values.reshape(28, 28), cmap="binary_r");

test_digito(1) # labels: 0
test_digito(3) # labels: 1
test_digito(5) # labels: 2
test_digito(7) # labels: 3
test_digito(2) # labels: 4
test_digito(0) # labels: 5
test_digito(13) # labels: 6
test_digito(15) # labels: 7
test_digito(17) # labels: 8
test_digito(4) # labels: 9


#%% Analizo cada dígito, creo DF para cada dígito
# cero
cero = df.copy()
cero = cero[cero['labels'] == 0]
cero.pop('labels')
cero.info()
cero.mean().mean()
cero.std().mean()
'''
Coeficiente de Variación (DS/Mean), lo calculo con el promedio del promedio de 
líneas y columnas y el promedio de la desviación estandar, con el fin de calcular 
la variación de la media respecto a la desviación standard, se calcula para cada
dígito
'''
cero.std().mean()/cero.mean().mean() 

#%% uno
uno = df.copy()
uno = uno[uno['labels'] == 1]
uno.pop('labels')
uno.info()
uno.mean().mean()
uno.std().mean()
uno.std().mean()/uno.mean().mean()

plt.imshow(uno.iloc[4].values.reshape(28, 28), cmap="binary_r"); # Pruebo con i = 0, 1, 2, 3, 4

#%% dos
dos = df.copy()
dos = dos[dos['labels'] == 2]
dos.pop('labels')
dos.info()
dos.mean().mean()
dos.std().mean()
dos.std().mean()/dos.mean().mean()

#%% tres
tres = df.copy()
tres = tres[tres['labels'] == 3]
tres.pop('labels')
tres.info()
tres.mean().mean()
tres.std().mean()
tres.std().mean()/tres.mean().mean()

#%% cuatro
cuatro = df.copy()
cuatro = cuatro[cuatro['labels'] == 4]
cuatro.pop('labels')
cuatro.info()
cuatro.mean().mean()
cuatro.std().mean()
cuatro.std().mean()/cuatro.mean().mean()

#%% cinco
cinco = df.copy()
cinco = cinco[cinco['labels'] == 5]
cinco.pop('labels')
cinco.info()
cinco.mean().mean()
cinco.std().mean()
cinco.std().mean()/cinco.mean().mean()

plt.imshow(cinco.iloc[3].values.reshape(28, 28), cmap="binary_r"); # Pruebo con i = 0, 1, 2, 3, 4

#%% seis
seis = df.copy()
seis = seis[seis['labels'] == 6]
seis.pop('labels')
seis.info()
seis.mean().mean()
seis.std().mean()
seis.std().mean()/seis.mean().mean()

#%% siete
siete = df.copy()
siete = siete[siete['labels'] == 7]
siete.pop('labels')
siete.info()
siete.mean().mean()
siete.std().mean()
siete.std().mean()/siete.mean().mean()

#%% ocho
ocho = df.copy()
ocho = ocho[ocho['labels'] == 8]
ocho.pop('labels')
ocho.info()
ocho.mean().mean()
ocho.std().mean()
ocho.std().mean()/ocho.mean().mean()

#%% nueve
nueve = df.copy()
nueve = nueve[nueve['labels'] == 9]
nueve.pop('labels')
nueve.info()
nueve.mean().mean()
nueve.std().mean()
nueve.std().mean()/nueve.mean().mean()

#%% ===========================================================================
# 2.-CLASIFICACIÓN BINARIA
#%% ===========================================================================

#%% a.- Contruyo nuevo DF escogiendo el dígito en la imágen
# Verifico imagen 0 e imagen 1
X = df.copy()
# X = X[X['labels'] == 0] # Pruebo con imagen 0
X = X[X['labels'] == 1] # Pruebo con imagen 1
y = X.pop('labels')

#%% Analizo nuevo DF X
X.info()
X.mean().mean()
X.std().mean()
X.std().mean()/X.mean().mean()
isna_validations(X) # Valido si tiene campos vacíos o valores nulls

plt.imshow(X.iloc[4].values.reshape(28, 28), cmap="binary_r"); # i = 0, 1, 2, 3, 4

#%% b.- Separar los datos en conjuntos de train y test (saltar a c para evaluar variación de atributos)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# Información de cada conjunto de datos
X_train.info()
X_test.info()
y_train.info()
y_test.info()

# Scale the features using StandardScaler
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%% c.- Construcción de subconjunto X: variación de atributos (luego separar los datos en el inciso "b")
# X = X.loc[:,['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']]
# X = X.loc[:,['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']]
X = X.loc[:,['0', '1', '2']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

#%% d.- Modelo KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=10)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#%% Metrics
P = model.predict_proba(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred, zero_division=np.nan)
recall = metrics.recall_score(y_test, y_pred, zero_division=np.nan)
accuracy

'''
Me causa curiosidad el por qué para todos los escenarios de modelos
me arroja siempre valor 1 en accuracy_score
'''

#%% ===========================================================================
# 3.-CLASIFICACIÓN MULTICLASE
#%% ===========================================================================
# Defino nuevamente el DF
X = df.copy()
y = X.pop('labels')

#%% separamos entre dev y eval
X_dev, X_eval, y_dev, y_eval = train_test_split(X,y,test_size=0.1, random_state = 20)
#%% experimento

'''
Hasta este punto logré avanzar en el TP
aún me falta continuar para ajustar y optimizar
esta parte del trabajo práctico
'''

alturas = list(range(1,21))
nsplits = 2
kf = KFold(n_splits=nsplits)

resultados = np.zeros((nsplits, len(alturas)))
# una fila por cada fold, una columna por cada modelo

for i, (train_index, test_index) in enumerate(kf.split(X_dev)):

    kf_X_train, kf_X_test = X_dev.iloc[train_index], X_dev.iloc[test_index]
    kf_y_train, kf_y_test = y_dev.iloc[train_index], y_dev.iloc[test_index]
    
    for j, hmax in enumerate(alturas):
        
        arbol = tree.DecisionTreeClassifier(max_depth = hmax)
        arbol.fit(kf_X_train, kf_y_train)
        pred = arbol.predict(kf_X_test)
        score = metrics.accuracy_score(kf_y_test,pred)
        
        resultados[i, j] = score
#%% promedio scores sobre los folds
scores_promedio = resultados.mean(axis = 0)


#%% 
for i,e in enumerate(alturas):
    print(f'Score promedio del modelo con hmax = {e}: {scores_promedio[i]:.4f}')
#%% entreno el modelo elegido en el conjunto dev entero
arbol_elegido = tree.DecisionTreeClassifier(max_depth = 1)
arbol_elegido.fit(X_dev, y_dev)
y_pred = arbol_elegido.predict(X_dev)

score_arbol_elegido_dev = metrics.accuracy_score(y_dev, y_pred)
print(score_arbol_elegido_dev)

#%% pruebo el modelo elegid y entrenado en el conjunto eval
y_pred_eval = arbol_elegido.predict(X_eval)       
score_arbol_elegido_eval = metrics.accuracy_score(y_eval, y_pred_eval)
print(score_arbol_elegido_eval)