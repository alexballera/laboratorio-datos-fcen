# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 10:09:48 2025

@author: ICBC
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsRegressor

#%% ALTURA MADRE VS HIJO VARON

ruta = 'G:/Mi unidad/uba/exactas/lab-datos/clases/Clase-17-RegresiónKNN/practica17/'

datos = pd.read_csv(ruta + "Resultados - Altura - 2025v - Alturas.csv")
datos = datos.head(37)
datos = datos.iloc[:,[0,1,2,3]]

datos_varones = datos[datos['Sexo al nacer (M/F)'] == 'M']
X = datos_varones[['altura madre']]

alturas = datos[['Altura (cm)']]
Y = alturas[datos['Sexo al nacer (M/F)'] == 'M']

plt.plot(X, Y, marker='.', color='k')
plt.title('', fontsize=14)
plt.xlabel('Altura madre', fontsize=12)
plt.ylabel('Altura hijo', fontsize=12)
plt.show()

#%%

neigh = KNeighborsRegressor(n_neighbors = 2)
neigh.fit(X, Y)

datonuevo = pd.DataFrame([{'altura madre': 156}]) # Predicción unica para madre con 156 cm
neigh.predict(datonuevo)

Y_pred = neigh.predict(X)
mean_squared_error(Y, Y_pred)
#%% Veamos cada K

res = []
for k in range(1,21):
    neigh = KNeighborsRegressor(n_neighbors = k)
    neigh.fit(X, Y)
   
    Y_pred = neigh.predict(X)
    error = mean_squared_error(Y, Y_pred)
    res.append((error, k))
   
#%% Graficamos los resultados

x = [t[1] for t in res]
y = [t[0] for t in res]

plt.plot(x, y, marker='.', color='k')

plt.title('', fontsize=14)
plt.xlabel('K', fontsize=12)
plt.ylabel('Mse', fontsize=12)
plt.xticks(x)

plt.grid(True)

# %%===========================================================================
# mpg
# =============================================================================

mpg = pd.read_csv(ruta + "auto-mpg.xls")

"""
mpg: miles per galon
displacement: Cilindrada

"""

print(mpg.dtypes)
#%%

X = pd.DataFrame(mpg['acceleration'])
Y = pd.DataFrame(mpg['mpg'])

modelo = KNeighborsRegressor(n_neighbors = 4)
modelo.fit(X, Y)

Y_pred = modelo.predict(X)

#%% Veamos cada K

res = []
for k in range(1,21):
    modelo = KNeighborsRegressor(n_neighbors = k)
    modelo.fit(X, Y)
   
    Y_pred = modelo.predict(X)
    error = mean_squared_error(Y, Y_pred)
    res.append((error, k))

#%% Graficamos los resultados

x = [t[1] for t in res]
y = [t[0] for t in res]

plt.plot(x, y, marker='.', color='k')

plt.title('', fontsize=14)
plt.xlabel('K', fontsize=12)
plt.ylabel('Mse', fontsize=12)
plt.xticks(x)

plt.grid(True)

#%% Pairplot
mpg.dtypes
sns.pairplot(mpg[['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year']])

#%% Veamos cada K

X = pd.DataFrame(mpg[['displacement', 'horsepower', 'weight']])
Y = pd.DataFrame(mpg['mpg'])

X['displacement_normalizado'] = ((X['displacement'] - X['displacement'].min()) / (X['displacement'].max() - X['displacement'].min()))
X['horsepower_normalizado'] =  ((X['horsepower'] - X['horsepower'].min()) / (X['horsepower'].max() - X['horsepower'].min()))
X['weight_normalizado'] = ((X['weight'] - X['weight'].min()) / (X['weight'].max() - X['weight'].min()))

X = X[['displacement_normalizado', 'horsepower_normalizado', 'weight_normalizado']]
Y = pd.DataFrame(mpg['mpg'])

res = []
for k in range(1,21):
    modelo = KNeighborsRegressor(n_neighbors = k)
    modelo.fit(X, Y)
   
    Y_pred = modelo.predict(X)
    error = mean_squared_error(Y, Y_pred)
    res.append((error, k))

#%% Graficamos los resultados

x = [t[1] for t in res]
y = [t[0] for t in res]

plt.plot(x, y, marker='.', color='k')

plt.title('', fontsize=14)
plt.xlabel('K', fontsize=12)

plt.ylabel('Mse', fontsize=12)
plt.xticks(x)

plt.grid(True)

#%%
X = pd.DataFrame(mpg[['displacement', 'horsepower', 'weight']])
Y = pd.DataFrame(mpg['mpg'])

X['displacement_normalizado'] = ((X['displacement'] - X['displacement'].min()) / (X['displacement'].max() - X['displacement'].min()))
X['horsepower_normalizado'] =  ((X['horsepower'] - X['horsepower'].min()) / (X['horsepower'].max() - X['horsepower'].min()))
X['weight_normalizado'] = ((X['weight'] - X['weight'].min()) / (X['weight'].max() - X['weight'].min()))

X = X[['displacement_normalizado', 'horsepower_normalizado', 'weight_normalizado']]
Y = pd.DataFrame(mpg['mpg'])

train = []
test = []
for k in range(1,20):
    x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.3) #70% para train y 30 para test

    modelo = KNeighborsRegressor(n_neighbors=k) # modelo en abstracto

    modelo.fit(x_train, y_train) # entreno el modelo con los datos

    y_pred_test = modelo.predict(x_test)
    y_pred_train = modelo.predict(x_train)
   
    error_test = mean_squared_error(y_test, y_pred_test)
    error_train = mean_squared_error(y_train, y_pred_train)

    train.append((error_train, k))
    test.append((error_test, k))

#%%

## TRAIN
x = [t[1] for t in train]
y = [t[0] for t in train]

plt.plot(x, y, marker='.', color='g', label='Train')

## TEST
x = [t[1] for t in test]
y = [t[0] for t in test]

plt.plot(x, y, marker='.', color='c', label='Test')

plt.title('K vs MSE', fontsize=12)
plt.xlabel('Número vecinos (k)', fontsize=10)
plt.ylabel('Mse', fontsize=10)
plt.xticks(x)
plt.legend()
plt.grid(True)
plt.show()


#%%

"""
   Ejercicio mpg kfolding
"""


mpg = pd.read_csv('auto-mpg.xls')

## Cargo mis datos
X = pd.DataFrame(mpg[['displacement', 'horsepower', 'weight']])
Y = pd.DataFrame(mpg['mpg'])

X['displacement_normalizado'] = ((X['displacement'] - X['displacement'].min()) / (X['displacement'].max() - X['displacement'].min()))
X['horsepower_normalizado'] =  ((X['horsepower'] - X['horsepower'].min()) / (X['horsepower'].max() - X['horsepower'].min()))
X['weight_normalizado'] = ((X['weight'] - X['weight'].min()) / (X['weight'].max() - X['weight'].min()))

X = X[['displacement_normalizado', 'horsepower_normalizado', 'weight_normalizado']]
Y = pd.DataFrame(mpg['mpg'])
###

#%% separamos entre dev y eval

X_dev, X_held_out, y_dev, y_eval = train_test_split(X, Y,test_size=0.1, random_state = 3)

#%%

rango = list(range(1,51))

nsplits = 10
kf = KFold(n_splits=nsplits)


resultados = np.zeros((nsplits, 50))
# una fila por cada fold, una columna por cada modelo

for i, (train_index, test_index) in enumerate(kf.split(X_dev)):

    kf_X_train, kf_X_test = X_dev.iloc[train_index], X_dev.iloc[test_index]
    kf_y_train, kf_y_test = y_dev.iloc[train_index], y_dev.iloc[test_index]
   
    for j, vecinos in enumerate(rango):
       
        modelo = KNeighborsRegressor(n_neighbors = vecinos)
        modelo.fit(kf_X_train, kf_y_train)
        pred = modelo.predict(kf_X_test)
        error = mean_squared_error(kf_y_test,pred)
       
        resultados[i, j] = error
       
#%% promedio scores sobre los folds
scores_promedio = resultados.mean(axis = 0)

#%%
for i,e in enumerate(rango):
    print(f'Score promedio del modelo con vecinos = {e}: {scores_promedio[i]:.4f}')
   
# Buscamos el minimo    
np.argmin(scores_promedio)
#%% 7 vecinos tiene el menor error

#%% entreno el modelo elegido en el conjunto dev entero
modelo_final = KNeighborsRegressor(n_neighbors = 7)
modelo_final.fit(X_dev, y_dev)
y_pred = modelo_final.predict(X_dev)

score_modelo_final_dev = mean_squared_error(y_dev, y_pred)
print(score_modelo_final_dev)

#%% pruebo el modelo elegido y entrenado en el conjunto eval
y_pred_eval = modelo_final.predict(X_held_out)      
score_modelo_final_dev = mean_squared_error(y_eval, y_pred_eval)
print(score_modelo_final_dev)