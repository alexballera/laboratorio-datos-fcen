# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 22:51:51 2025

@author: ICBC
"""
import random
import numpy as np
import pandas as pd

# Ejercicio 1
def generala_tirar():
    numeros = []
    for i in range(0,5):
        numeros.append(random.randint(1,6))
    return numeros

# print("Ejercicio 1:", generala_tirar())

# Ejercicio 2
def diccionario_repeticiones(lista):
    repeticiones = {}
    for i in lista:
        if i in repeticiones:
            repeticiones[i] += 1
        else:
            repeticiones[i] = 0
    return repeticiones


def generala_tirar_opciones(lista):
    repeticiones = diccionario_repeticiones(lista)
    escalera = [[1,2,3,4,5], [2,3,4,5,6], [3,4,5,6,1]]
    texto = 'Ninguna de las anteriores'

    for i in escalera:
        if i == lista:
            texto = 'Escalera'

    for v in repeticiones:
        repeticiones[v] += 1

    for i in repeticiones:
        if repeticiones[i] == 2:
            for n in repeticiones:
                if repeticiones[n] == 3:
                    texto = 'Full'
        if repeticiones[i] == 4:
            texto = 'Poker'
        if repeticiones[i] == 5:
            texto = 'Generala'
    return lista, texto


escalera = [2,3,4,5,6]
full = [2,3,3,2,2]
poker = [1,1,1,1,4]
generala = [1,1,1,1,1]
ninguna = [1,2,1,2,5]
aleatorio = generala_tirar()
# print(generala_tirar_opciones(aleatorio))

# Ejercicio 3
def solo_palabra(palabra):
    with open('datame.txt', 'rt') as file:
        arr = []
        for linea in file:
            arr.append(linea)
        nuevo_arr = []
        for item in arr:
            if palabra in item:
                nuevo_arr.append(item)
        return nuevo_arr
    

palabra = "estudiantes"
# print(solo_palabra(palabra))

# Ejercicio 4
def lista_materias():
    with open('cronograma_sugerido.csv', 'rt') as file:
        res = []
        next(file)
        for linea in file:
            res.append(linea.split(',')[1])
        return res

# print(lista_materias())

# Ejercicio 5
def cuantas_materias(n):
    if n < 3 or n > 8:
        return 'Ha introducido un dato erróneo, debe ser entre 3 y 8'
    with open('cronograma_sugerido.csv', 'rt') as file:
        res = []
        next(file)
        for linea in file:
            res.append(linea.split(',')[0])
        i = 0
        for num in res:
            if num == str(n):
                i += 1
        return i

# print(cuantas_materias(5))

# Ejercicio 6
def materias_cuatrimestre(nombre_archivo, n):
    materias = []
    materia = {}

    with open(nombre_archivo, 'rt') as file:
        encabezado = next(file).split(',')
        for linea in file:
            linea = linea.split(',')

            if linea[0] == str(n):
                cuatrimestre = linea[0]
                asignatura = linea[1].split('\n')[0]
                correlativa = linea[2].split('\n')[0]
                head_cuatrimestre = encabezado[0]
                head_asignatura = encabezado[1]
                head_correlativa = encabezado[2].split('\n')[0]
                materia[head_cuatrimestre] = cuatrimestre
                materia[head_asignatura] = asignatura
                materia[head_correlativa] = correlativa

                materia_copy = materia.copy()
                materias.append(materia_copy)

    return materias

materias = materias_cuatrimestre('cronograma_sugerido.csv', 3)
# print(materias)

# Ejercicio Pag 47 MAtriz con Numpy
def pisar_elemento(M,e):
    M_COPY = np.copy(M)
    i = 0
    j = 0
    for linea in M:
        for col in linea:
            if col == e:
                M_COPY[i,j] = -1
            j += 1
        i += 1
    return M_COPY

M = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
e = 2
Ma = pisar_elemento(M,e)

# Dataframe: Crear un Data Frame con un array
def crear_data_frame_desde_array(arr):
    
    M = np.array(arr)
    pd2 = pd.DataFrame(M, columns = ['a', 'b', 'c', 'd'], index = ['v1','v2','v3'])
    
    return pd2

arr = [[11, 1, -5, 3],[10, 5, 6, 7],[3, 8, 10, -1]]
pd2 = crear_data_frame_desde_array(arr)

# Cargar un Data Frame desde un archivo
def cargar_un_data_frame_desde_archivo(nombre_archivo):
    pd2 = pd.read_csv(nombre_archivo)
    
    return pd2
    
nombre_archivo = 'cronograma_sugerido.csv'
pd3 = cargar_un_data_frame_desde_archivo(nombre_archivo)

# Cargar arbolado en espacios verdes
def cargar_arbolado(arbolado):
    df = pd.read_csv(arbolado)    
    
    df_jacarandas = df[df['nombre_com'] == 'Jacarandá']        
    
    df_palo_borracho = df[df['nombre_com'] == 'Palo borracho']
    
    # Cantidad
    print('Cantidad total', len(df))
    print('Cantidad jacarandas', len(df_jacarandas))
    print('Cantidad palo_borracho', len(df_palo_borracho))

    print()
    
    # Altura máxima
    print('Altura máxima todos', df['altura_tot'].max())
    print('Altura máxima jacarandas', df_jacarandas['altura_tot'].max())
    print('Altura máxima palo_borracho', df_palo_borracho['altura_tot'].max())

    print()
    
    # Altura mínima
    print('Altura mínima todos', df['altura_tot'].min())
    print('Altura mínima jacarandas', df_jacarandas['altura_tot'].min())
    print('Altura mínima palo_borracho', df_palo_borracho['altura_tot'].min())

    print()
    
    # Diámetro máximo
    print('Diámetro máxima todos', df['diametro'].max())
    print('Diámetro máxima jacarandas', df_jacarandas['diametro'].max())
    print('Diámetro máxima palo_borracho', df_palo_borracho['diametro'].max())

    print()
    
    # Diámetro mínimo
    print('Diámetro mínimo todos', df['diametro'].min())
    print('Diámetro mínimo jacarandas', df_jacarandas['diametro'].min())
    print('Diámetro mínimo palo_borracho', df_palo_borracho['diametro'].min())

    print()
    
    # Diámetro promedio
    print('Diámetro promedio todos', round(df['diametro'].mean(), 2))
    print('Diámetro promedio jacarandas', round(df_jacarandas['diametro'].mean(), 2))
    print('Diámetro promedio palo_borracho', round(df_palo_borracho['diametro'].mean(), 2))
    
    
    return df

arbolado = 'arbolado-en-espacios-verdes.csv'
arbolado_df = cargar_arbolado(arbolado)
