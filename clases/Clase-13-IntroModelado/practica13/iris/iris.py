# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:16:28 2025

@author: ICBC
"""
#%% Imports
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import tree
import pandas as pd

#%% Imports

iris = load_iris(as_frame = True)

data = iris.frame
atributos = iris.dataY = iris.target

iris.target_names
diccionario = dict(zip([0,1,2], iris.target_names))
atri = ['sepal length (cm)', 'sepal width (cm)', 'portal length (cm)', 'petal width (cm)']

#%% --- Graficos
nbins = 35
f, s = plt.subplots(2,2)
plt.suptitle('Histograma de los 4 atributos', size = 'large')

sns.histplot(data = data, x = 'sepal length (cm)', hue = 'target', bins = nbins, stat = 'probability', ax=s[0,0], palette = 'viridis')


sns.histplot(data = data, x = 'sepal width (cm)', hue = 'target', bins = nbins, stat = 'probability', ax=s[0,1], palette = 'viridis')


sns.histplot(data = data, x = 'petal length (cm)', hue = 'target', bins = nbins, stat = 'probability', ax=s[1,0], palette = 'viridis')


sns.histplot(data = data, x = 'petal width (cm)', hue = 'target', bins = nbins, stat = 'probability', ax=s[1,1], palette = 'viridis')

#%% ---
umbrales = [1,2,3,4]
umbral = umbrales[0]

for umbral in

data_clasif['clase_asignada'] = atributos.apply(lambda row: clasificador_iris(row), axis= 1)

def clasificador_iris(fila):
    pet_l = fila['petal length (cm)']
    if pet_l < 2.5:
        case = 0
    elif pet_l < umbral:
        case = 1
    else:
        case = 2
#%% ---
clases = set(data['target'])
matriz_confusion = np.zeros((3,3))

for i in range(3):
    for j in range(3):
        filtro = (data_clasif['target'] == i) & (data_clasif['clase_asignada'] == j)
        cuenta = len(data_clasif[filtro])
        matriz_confusion[i, j] = cuenta
        
matriz_confusion

#%% ---
exacti = sum(data_clasif['target'])
data_clasif['clase_asignada']
