# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:45:25 2025

@author: ICBC
"""
#%% Imports
import pandas as pd
import duckdb as dd

#%% datasets
df = pd.read_csv('titanic_training.csv')

#%% análisis sex
res = dd.sql("""
             SELECT Sex, Survived
             FROM df
             GROUP BY Sex, Survived
             """).df()

#%% 



