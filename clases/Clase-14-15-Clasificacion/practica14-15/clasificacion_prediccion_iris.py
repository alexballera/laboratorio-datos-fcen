# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""
#%% Imports
import pandas as pd
import duckdb as dd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


#%% Obtengo y preparo los datos
iris = load_iris(as_frame = True)
X = iris.frame
y = iris.target

#%%
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X, y)

prediccion = model.predict(X)

comparacion = X.iloc[:,[4]] # creo un dataframe con la columna 5 (target)

comparacion['prediction'] = prediccion

#%% Constructor
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.3)

model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

print('Exactitud del modelo: ', metrics.accuracy_score(Y_test, Y_pred))
