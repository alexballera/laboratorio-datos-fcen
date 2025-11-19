# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 10:46:33 2025

@author: ICBC
"""

import pandas as pd
#%% Tiempos de reacción con diferentes manos
df = pd.read_csv('datos_tiempo_reaccion.csv')
#%% 
#%% Tiempos de reacción con habil / no habil
df = pd.read_csv('DatosTiemposDeReaccion-HabilNoHabil.csv')

description = df.describe()

mean = df.mean()
ds = df.std()

#%% 