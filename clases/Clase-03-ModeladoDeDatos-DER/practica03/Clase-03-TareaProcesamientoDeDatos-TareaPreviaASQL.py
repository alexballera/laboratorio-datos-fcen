# -*- coding: utf-8 -*-
import numpy as np

def superanSalarioActividad01(empleados, n):
    res = []
    for row in empleados:
        if row[3] > n:
            res.append(row)
    return res
    
empleado_01 = np.array([[20222333,45,2,20000], [33456234,40,0,25000], [45432345,41,1,10000]])
empleado_02 = np.array([[20222333,45,2,20000], [33456234,40,0,25000], [45432345,41,1,10000], [43967304,37,0,12000], [42236276,36,0,18000]])
empleado_03 = np.array([[20222333,20000,45,2], [33456234,25000,40,0], [45432345,10000,41,1], [43967304,12000,37,0], [42236276,18000,36,0]])
n = 15000
empleado_04 = np.array([[20222333,33456234,45432345,43967304,42236276], [20000,25000,10000,12000,18000], [45,41,40,37,36], [2,0,1,0,0]])
n = 15000

superan_15000 = superanSalarioActividad01(empleado_01, n)
superan_15000_2 = superanSalarioActividad01(empleado_02, n)
superan_15000_3_a = superanSalarioActividad01(empleado_03, n)
superan_15000_4_a = superanSalarioActividad01(empleado_04, n)

def superanSalarioActividad03(empleados, n):
    res = []
    for row in empleados:
        nuevo_row = []
        if row[1] > n:
            nuevo_row.append(row[0].tolist())
            nuevo_row.append(row[2].tolist())
            nuevo_row.append(row[3].tolist())
            nuevo_row.append(row[1].tolist())
            res.append(nuevo_row)
    return res

superan_15000_3_b = superanSalarioActividad03(empleado_03, n)
superan_15000_4_b = superanSalarioActividad03(empleado_04, n)

def superanSalarioActividad04(empleados, n):
    if len(empleados) == 0:
        return []

    result = [[None for i in range(len(empleados))] for j in range(len(empleados[0]))]

    for i in range(len(empleados[0])):
        for j in range(len(empleados)):
            result[i][j] = empleados[j][i]

    return result

superan_15000_4_c = superanSalarioActividad03(superanSalarioActividad04(empleado_04, n), n)
transpuesta = empleado_04.T
# =============================================================================
# 1.-¿Cómo afectó?
#   a.-Al agregar más filas
#       Al agregar más filas no afectan las funciones desarrolladas, el resultado dependería de los nuevos datos
#   b.-En caso de alterar el orden de las columnas
#       Afecta el resultado y hay que modificar el código desarrollado, ya que el programa realiza las búsquedas por filas y por el filtro que se realiza por una columna
# 2.-De filas a columnas
#   Afecta considerablemente ya que se estandariza tener los datos por filas y las columnas representan las variables, en este caso hay que transponer la matriz para poder recorrerla correctamente
# 3.-Al disponer del modelo de datos previamente le facilita al desarrollador realizar el programa acorde a las especificaciones y así cumplir con el contrato
# =============================================================================
