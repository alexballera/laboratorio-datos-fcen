# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""
#%% Imports
import pandas as pd
import duckdb as dd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

#%% Constructor
df = pd.read_csv('titanic_training.csv')

# y atributo a predecir - valor actual
# y = dd.sql("""
#            SELECT Survived
#            FROM df
#            """).df()

y = df['Survived']

# X atributos a considerar en el modelo predictor
X = dd.sql("""
            SELECT  PassengerId,
                    
                    Pclass,
                    CASE WHEN Sex = 'male'
                         THEN 1
                         ELSE 0
                    END AS Sex,
                    Age
            FROM df
            """).df()

#%% ---
arbol = DecisionTreeClassifier(max_depth=5)
arbol.fit(X, y) # Entrenamiento del modelo
#%% --- Visualización
tree.plot_tree(arbol)
#%% --- Prediccion
prediction = arbol.predict(X)
#%% --- Comparación
comparacion = df.iloc[:,[1]]
comparacion['prediccion'] = prediction

aciertos = dd.sql("""
                  SELECT COUNT(*)
                  FROM comparacion
                  WHERE Survived = prediccion
                  """).df()
total = df['PassengerId'].count()    
aciert = (aciertos / total) * 100
