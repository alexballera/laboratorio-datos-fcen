# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 18:14:53 2025

@author: Admin
"""
#%% KNN --->>> K = cantidad de vecinos cercanos
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

#%% creamos dataframes

clientes = pd.read_csv('creditos.csv')
solventes = clientes[clientes['cumplio'] == 1]
riesgosos = clientes[clientes['cumplio'] == 0]

#%% visualizamos

plt.scatter(solventes['edad'], solventes['credito'],
            marker='*',
            s=150,
            color='skyblue',
            label='Si pagó (Clase: 1')

plt.scatter(riesgosos['edad'], riesgosos['credito'],
            marker='*',
            s=150,
            color='red',
            label='No pagó (Clase: 0')

plt.ylabel('Monto del crédito')
plt.xlabel('Edad')
plt.legend(bbox_to_anchor=(1, 0.2))
plt.show()

#%% Preparación de los datos (escalar): procesar, standarizar, escalar, normalizar los datos

datos = clientes[['edad', 'credito']] # datos
Y = clientes['cumplio'] # clase

escalador = preprocessing.MinMaxScaler() # escala los datos en 0 a 1

X = escalador.fit_transform(datos)

#%% Creacion del Modelo KNN, n_neighbors = K = cantidad de vecinos cercanos

clasificador = KNeighborsClassifier(n_neighbors=3)
clasificador.fit(X, Y)

#%% Verificamos con un cliente
edad = 50
monto = 350000

# Escalamos los datos del nuevo solicitante
solicitante = escalador.transform([[edad, monto]])

# Calcular clase y probabilidades
print('Clase', clasificador.predict(solicitante))
print('Probabilidad por clase', clasificador.predict_proba(solicitante))

# visualizamos

plt.scatter(solventes['edad'], solventes['credito'],
            marker='*',
            s=150,
            color='skyblue',
            label='Si pagó (Clase: 1')

plt.scatter(riesgosos['edad'], riesgosos['credito'],
            marker='*',
            s=150,
            color='red',
            label='No pagó (Clase: 0')

plt.scatter(edad,
            monto,
            marker='P',
            s=150,
            color='green',
            label='Solicitante'
            )

plt.ylabel('Monto del crédito')
plt.xlabel('Edad')
plt.legend(bbox_to_anchor=(1, 0.3))
plt.show()

#%% Regiones de clases: pagadores vs deudores
#Datos sinténticos de todos los posibles solicitantes
creditos = np.array([np.arange(100000, 600010, 1000)]*43).reshape(1, -1)
edades = np.array([np.arange(18, 61)]*501).reshape(1, -1)
todos = pd.DataFrame(np.stack((edades, creditos), axis=2)[0],
                     columns=["edad", "credito"])

#Escalar los datos
solicitantes = escalador.transform(todos)

#Predecir todas las clases
clases_resultantes = clasificador.predict(solicitantes)

#Código para graficar
buenos = todos[clases_resultantes==1]
malos = todos[clases_resultantes==0]
plt.scatter(buenos["edad"], buenos["credito"],
            marker="*", s=150, color="skyblue", label="Sí pagará (Clase: 1)")
plt.scatter(malos["edad"], malos["credito"],
            marker="*", s=150, color="red", label="No pagará (Clase: 0)")
plt.ylabel("Monto del crédito")
plt.xlabel("Edad")
plt.legend(bbox_to_anchor=(1, 0.2))
plt.show()