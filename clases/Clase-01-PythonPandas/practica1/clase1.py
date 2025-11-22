#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Alex Ballera
"""
# %% Copias
a = [1, 2, 3, 4, 5]
b = a  # Copia por referencia
c = a.copy()  # Copia por valor
d = a[:]  # Copia por valor
print("Lista original a:", a)
print("Copia por referencia b:", b)
print("Copia por valor c:", c)
print("Copia por valor d:", d)
# %%
print("Antes de modificar a:")
print(b == a)
print(b is a)
print(c == a)
print(c is a)
# %%
a.append(6)
print("Después de modificar a:")
print("Lista original a:", a)
print("Copia por referencia b:", b)
print("Copia por valor c:", c)
print("Copia por valor d:", d)
# %%
import math
print(math.sqrt(2))
print(math.exp(2))
print(math.cos(120))
print(math.log(8))
print(math.factorial(5))
print(math.gcd(48, 18))
# %%
ruta = './'
nombre_archivo = 'datame.txt'
f = open(ruta + nombre_archivo, 'rt')
data = f.read()
f.close()
print(data)
# %%
with open(ruta + nombre_archivo, 'rt') as f:
    data = f.read()
print(data)
# %%
data_nuevo = 'Línea nueva\n \nOtra línea nueva\n\n' + data
data_nuevo = data_nuevo + '\nÚltima línea nueva\n'

datame = open('datame_modificado.txt', 'w')
datame.write(data_nuevo)
datame.close()
#
# %%