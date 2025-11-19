# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 09:20:23 2025

@author: Admin
"""

import pandas as pd
import duckdb as dd

#%% 2024 1C

colectivo = {'linea': [42, 42, 107, 107, 10], 'ramal': ['A', 'B', 'A', 'B', 'A'], 'cantidad': [30, 40, 28, 22, 20]}
terminal = {'linea': [42, 42, 107, 107, 10, 10], 'ramal': ['A', 'B', 'A', 'B', 'A', 'B'], 'cabecera': ['River', 'Ciudad Universitaria', 'Ciudad Universitaria', 'Ramsay', 'Palermo', 'Aeroparque']}

colectivo = pd.DataFrame(colectivo)
terminal = pd.DataFrame(terminal)

#%% consulta 1

res = dd.sql("""
                   SELECT c.linea, c.ramal, t.cabecera
                   FROM colectivo AS c
                   LEFT OUTER JOIN terminal AS t
                   ON c.linea = t.linea
                   WHERE c.linea < 100
                   ORDER BY c.linea ASC, c.ramal ASC, t.cabecera DESC
                   """).df()

#%% consulta 2

res = dd.sql("""
                   SELECT c.linea, COUNT(*) AS TotalRamales, SUM(c.cantidad) AS TotalUnidades
                   FROM colectivo AS c
                   INNER JOIN terminal AS t
                   ON c.linea = t.linea AND c.ramal = t.ramal
                   GROUP BY c.linea
                   HAVING TotalUnidades < 60
                   ORDER BY c.linea ASC
                   """).df()

#%% 2024 2C

sucursales = {'n_sucursal': [123,52,107,141], 'barrio': ['Palermo', 'Villa Crespo', 'Belgrano', 'Palermo'], 'cant_cajeros': [2,4,3,3], 'cant_cajas': [3,3,2,4]}
zonas = {'barrio': ['Belgrano', 'Caballito', 'Flores', 'Palermo', 'Palermo'], 'zona': ['Norte', 'Sur', 'Sur', 'Norte', 'Sur'], 'ciudad': ['CABA', 'CABA', 'CABA', 'CABA', 'CABA'], 'codigo_postal': [1411,1406,1321,1418,1425]}

sucursales = pd.DataFrame(sucursales)
zonas = pd.DataFrame(zonas)

#%% consulta 1
res = dd.sql("""
             SELECT s.n_sucursal, s.barrio, z.codigo_postal
             FROM sucursales AS s
             LEFT JOIN zonas AS z
             ON s.barrio = z.barrio
             ORDER BY s.n_sucursal DESC
             """).df()
#%% consulta 2
res = dd.sql("""
             SELECT s.barrio, COUNT(*) AS TotalSucursales, SUM(s.cant_cajeros) AS TotalCajeros
             FROM sucursales AS s
             INNER JOIN zonas AS z
             ON s.barrio = z.barrio
             GROUP BY s.barrio
             HAVING TotalCajeros > 3
             ORDER BY s.barrio ASC
             """).df()
#%% 2023 1C Parcial

turismo = {'atractivo': ['Cataratas', 'La Quiaca', 'Camino de Santiago'], 'pais': ['Argentina', 'Argentina', 'España']}
ubicacion = {'pais': ['Argentina', 'Argentina', 'España'], 'provincia': ['Misiones', 'Jujuy', 'La Rioja'], }

turismo = pd.DataFrame(turismo)
ubicacion = pd.DataFrame(ubicacion)

#%% consulta 1

res = dd.sql("""
             SELECT a.atractivo, u.pais, u.provincia
             FROM turismo AS a
             INNER JOIN ubicacion AS u
             ON a.pais = u.pais
             """).df()

#%% consulta 2

res = dd.sql("""
             SELECT a.pais, COUNT(*) AS total
             FROM turismo AS a
             WHERE pais LIKE '%_a'
             GROUP BY pais
             HAVING total >= 2
             """).df()

#%% 2023 1C Recuperatorio

empleado = {'tipo_doc': ['DNI', 'DNI', 'DNI', 'DNI'], 'nro': ['12345678', '23456789', '34567890', '45678901'], 'nombre': ['Juan Perez', 'Maria Gonzalez', 'Alejandro Rodriguez', 'Lucia Martinez'], 'fecha_ingreso': ['2021-05-10', '2022-01-15', '2020-11-30', '2023-03-18'], 'id_proyecto': [2,4,2,3]}
proyecto = {'id_proyecto': [1,2,2,4], 'id_subproyecto': ['A', 'A', 'H', 'C'], 'nombre_proy': ['PETROLEO', 'PANDAS', 'BALLENAS', 'CO2']}

empleado = pd.DataFrame(empleado)
proyecto = pd.DataFrame(proyecto)

#%% consulta 1
res = dd.sql("""
             SELECT e.tipo_doc, e.nro, e.nombre, p.nombre_proy
             FROM empleado AS e
             LEFT OUTER JOIN proyecto AS p
             ON e.id_proyecto = p.id_proyecto
             WHERE e.id_proyecto <> 5
             """).df()

#%% consulta 2
res = dd.sql("""
             SELECT nombre
             FROM empleado
             WHERE fecha_ingreso = (
                 SELECT MIN(fecha_ingreso)
                 FROM empleado
                 )
             """).df()