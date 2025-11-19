# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 21:48:27 2025

@author: Admin
"""
#%% Datos

y_verdadero = [1, 2, 3, 4, 5]
y_predicho = [1, 2, 3, 4, -5]

#%% Error absoluto máximo (M)
from sklearn.metrics import max_error
max_error(y_verdadero, y_predicho)

#%% Error absoluto medio (mean absolute error - MAE)
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_verdadero, y_predicho)

#%% Error cuadrático medio (mean squared error - MSE)
from sklearn.metrics import mean_squared_error
mean_squared_error(y_verdadero, y_predicho)

#%% Suma de los cuadrados de los residuos (RSS)
mean_squared_error(y_verdadero, y_predicho)*len(y_predicho)