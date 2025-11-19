import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error

# %%===========================================================================
# roundup
# =============================================================================
ru = pd.read_csv("datos_libreta_68824.txt", delim_whitespace=' ')

# %% Aproximar recta
# Y = a + b*X

X = np.linspace(min(ru['RU']), max(ru['RU']))
x = ru.iloc[:,[0]]
y = ru.iloc[:,[1]]
y1 = y.iloc[0]
y2 = y.iloc[1]
x1 = x.iloc[0]
x2 = x.iloc[1]
a = 105.26
b = (y2.iloc[0] - y1.iloc[0]) / (x2.iloc[0] - x1.iloc[0])

print(a,b)

Y = a + b*X

plt.scatter(ru['RU'], ru['ID'])
plt.plot(X, Y,  'r')
plt.show()

#%% Obtener recta de cuadrados minimos
b, a = np.polyfit(ru['RU'], ru['ID'], 1)
print(a,b)
Y = a + b*X
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

print(mpg.dtypes)

# %% Comparar variables con graficos

def versus(col1, col2):
    
    plt.show()

#%% Comparar variables y calcular recta de cuadrados minimos

def reg_lineal(col1, col2, grado=1):
    
    plt.show()
    
#%% Comparar variables, calcular recta de cuadrados minimos y calcular R²

def reg_lineal_r2(col1, col2, grado=1):
    
    plt.title("R²: " + str(r2))
    plt.show()