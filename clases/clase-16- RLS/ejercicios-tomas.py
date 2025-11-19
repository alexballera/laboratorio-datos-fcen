# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 09:25:02 2025

@author: ICBC
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsRegressor
# %%===========================================================================
# roundup
# =============================================================================
ru = pd.read_csv("datos_roundup.txt", delim_whitespace=' ')

# %% Aproximar recta
# Y = a + b*X

X = np.linspace(min(ru['RU']), max(ru['RU']))
a = 106.5
b = 0.037
Y = a + b*X

plt.scatter(ru['RU'], ru['ID'])
plt.plot(X, Y,  'r')
plt.show()

#%% Obtener recta de cuadrados minimos

b, a = np.polyfit(ru['RU'], ru['ID'], 1)
Y = a + b*X
plt.scatter(ru['RU'], ru['ID'])
plt.plot(X, Y, 'k')
plt.show()

#%% Calcular score R²
X = ru['RU']
Y = ru['ID']
Y_pred = a + b*X

r2 = r2_score(Y, a + b*X)
print("R²: " + str(r2))

mse = mean_squared_error(Y, Y_pred)
print("MSE: " + str(mse))
#%%
ru = pd.read_csv("datos_libreta_26223.txt", delim_whitespace=' ')

b, a = np.polyfit(ru['RU'], ru['ID'], 1)

#Calcular score R²
X = ru['RU']
Y = ru['ID']
Y_pred = a + b*X

r2 = r2_score(Y, a + b*X)
print("R²: " + str(r2))

mse = mean_squared_error(Y, Y_pred)
print("MSE: " + str(mse))

print(a)
print(b)

# %%===========================================================================
# Anascombe
# =============================================================================
df = sns.load_dataset("anscombe")


# %%===========================================================================
# mpg
# =============================================================================

mpg = pd.read_csv("auto-mpg.xls")

"""
mpg: miles per galon
displacement: Cilindrada

"""

print(mpg.dtypes)

# %% Comparar variables con graficos

def versus(col1, col2):
    b, a = np.polyfit(col1, col2, 1)
    plt.scatter(col1, col2)
    plt.show()

versus(mpg.mpg, mpg.horsepower)
#%% Comparar variables y calcular recta de cuadrados minimos

def reg_lineal(col1, col2, grado=1):
    X = np.linspace(min(col1), max(col1))
   
    b, a = np.polyfit(col1, col2, grado)
    plt.scatter(col1, col2)
    plt.plot(X, a + b*X, color = 'k')
    plt.show()

reg_lineal(mpg.mpg, mpg.horsepower)
reg_lineal(mpg.weight, mpg.horsepower)    

#%% Comparar variables y calcular recta de cuadrados minimos

def reg_cuadratica(col1, col2, grado=2):
    X = np.linspace(min(col1), max(col1))
   
    c, b, a = np.polyfit(col1, col2, grado)
    plt.scatter(col1, col2)
    plt.plot(X, a + b*X + c*X**2, color = 'k')
    plt.show()

reg_cuadratica(mpg.mpg, mpg.horsepower)
reg_cuadratica(mpg.displacement, mpg.weight)  

#%% Comparar variables, calcular recta de cuadrados minimos y calcular R²

def reg_lineal_r2(col1, col2, grado=1):
    X = col1
    Y = col2
    b, a = np.polyfit(col1, col2, grado)
    plt.scatter(col1, col2)
    plt.plot(X, a + b*X, color = 'k')
    r2 = r2_score(Y, a + b*X)
    plt.title("R²: " + str(r2))
    plt.show()

reg_lineal_r2(mpg.weight, mpg.displacement)  
reg_lineal_r2(mpg.weight, mpg.mpg)  
reg_lineal_r2(mpg.displacement, mpg.weight)  

#%%

from sklearn.neighbors import KNeighborsRegressor

datos = pd.read_csv("Resultados - Altura - 2025v - Alturas.csv")
datos = datos.head(37)
datos = datos.iloc[:,[0,1,2,3]]

datos_varones = datos[datos['Sexo al nacer (M/F)'] == 'M']
X = datos_varones[['altura madre']]

alturas = datos[['Altura (cm)']]
Y = alturas[datos['Sexo al nacer (M/F)'] == 'M']

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

#%%


# CAMBIAMOS AL DATASET DE MPG-AUTO


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

sns.pairplot(mpg[['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration']])

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
from sklearn.model_selection import train_test_split

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

plt.plot(x, y, marker='.', color='k')

plt.title('', fontsize=14)
plt.xlabel('K', fontsize=12)
plt.ylabel('Mse', fontsize=12)
plt.xticks(x)

plt.grid(True)

## TEST
x = [t[1] for t in test]
y = [t[0] for t in test]

plt.plot(x, y, marker='.', color='c')

plt.title('', fontsize=14)
plt.xlabel('K', fontsize=12)
plt.ylabel('Mse', fontsize=12)
plt.xticks(x)

plt.grid(True)


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