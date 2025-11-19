# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 22:39:39 2025

@author: ICBC
"""
#%% Imports
import pandas as pd

#%% ---
arbolado = pd.read_csv('arbolado-en-espacios-verdes.csv')

col = arbolado['long']
row_loc = arbolado.loc[0]
row_i = arbolado.iloc[0]
print(row_loc == row_i)
#%%