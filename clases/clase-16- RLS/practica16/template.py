import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error

# %%===========================================================================
# roundup
# =============================================================================
ru = pd.read_csv("datos_roundup.txt", delim_whitespace=' ')

# %% Aproximar recta
# Y = a + b*X

X = np.linspace(min(ru['RU']), max(ru['RU']))
x = ru['ID']
y = ru['RU']
y1 = y.iloc[0]
y2 = y.iloc[1]
x1 = x.iloc[0]
x2 = x.iloc[1]
a = 104
b = (y2 - y1) / (x2 - x1)
Y = a + b*X

b = 0.037
a = 106.5
Y = a + b*X

plt.scatter(ru['RU'], ru['ID'])
plt.plot(X, Y,  'r')
plt.show()

#%% Obtener recta de cuadrados minimos
b, a = np.polyfit(ru['RU'], ru['ID'], 1)

plt.scatter(ru['RU'], ru['ID'])
plt.plot(X, Y, 'k')
plt.show()

#%% Calcular score R²

Y = ru['ID']
X = ru['RU']
Y_pred = a + b*X

r2 = r2_score(Y, Y_pred)
print("R²: " + str(r2))


mse = mean_squared_error(Y, Y_pred)
print("MSE: " + str(mse)) # Promedio error cuadrado (error esperado)

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

col1 = mpg['weight']
col2 = mpg['displacement']
print(mpg.dtypes)

# %% Comparar variables con graficos

def versus(col1, col2):
    plt.scatter(col1, col2)
    plt.show()

versus(col1, col2)

#%% Comparar variables y calcular recta de cuadrados minimos

def reg_lineal(col1, col2, grado=1):
    b, a = np.polyfit(col1, col2, 1)
    X = np.linspace(min(col1), max(col1))
    Y = a + b*X
    plt.plot(X, Y, 'k')
    versus(col1, col2)
    
    plt.show()
reg_lineal(col1, col2)
#%% Comparar variables, calcular recta de cuadrados minimos y calcular R²

def reg_lineal_r2(col1, col2, grado = 1):

    b, a = np.polyfit(col1, col2, grado)
    X = np.linspace(min(col1), max(col1))
    Y = a + b*X

    Y_pred = a + b*col1
    r2 = r2_score(col2, Y_pred)

    plt.scatter(col1, col2) # grafica de puntos
    plt.plot(X, Y, 'k') # grafica lineal
    plt.title("R²: " + str(r2))
    plt.show()

reg_lineal_r2(mpg['weight'], mpg['horsepower'])

#%% reg_cuadratica_r2
def reg_cuadratica_r2(col1, col2, grado):

    c, b, a = np.polyfit(col1, col2, grado)
    X = np.linspace(min(col1), max(col1))
    Y = a + b*X + c*X**2

    Y_pred = a + b*col1 + c*col1**2
    r2 = r2_score(col2, Y_pred)

    plt.scatter(col1, col2) # grafica de puntos
    plt.plot(X, Y, 'k') # grafica lineal
    plt.title("R²: " + str(r2))
    plt.show()

reg_cuadratica_r2(mpg['weight'], mpg['horsepower'], 2)
