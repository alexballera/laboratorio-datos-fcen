# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 20:08:09 2025
Guía Práctica - SQL

@author: Alexander Ballera
"""

# Importamos bibliotecas
import pandas as pd
import duckdb as dd
import numpy as np


#%%===========================================================================
#%% Importamos los datasets que vamos a utilizar en este programa
#=============================================================================

casos = pd.read_csv('casos.csv')
departamento = pd.read_csv('departamento.csv')
grupoetario = pd.read_csv('grupoetario.csv')
provincia = pd.read_csv('provincia.csv')
tipoevento = pd.read_csv('tipoevento.csv')

#===========================================================================
# A. Consultas sobre una tabla
#===========================================================================

#%% a. Listar sólo los nombres de todos los departamentos que hay en la tabla departamento (dejando los registros repetidos).
consultaSQL = """
                SELECT descripcion
                FROM departamento
              """
resultado = dd.sql(consultaSQL).df()

res = departamento['descripcion']
res = pd.DataFrame(res)

set(resultado) == set(res)

#%%-----------
# b. Listar sólo los nombres de todos los departamentos que hay en la tabla departamento (eliminando los registros repetidos).

consultaSQL = """
                SELECT DISTINCT descripcion
                FROM departamento
              """
resultado = dd.sql(consultaSQL).df()

res = departamento['descripcion']
res = pd.DataFrame(res)
res = res.drop_duplicates()

set(resultado) == set(res)

#%%-----------
# c. Listar sólo los códigos de departamento y sus nombres, de todos los departamentos que hay en la tabla departamento.

consultaSQL = """
                SELECT id, descripcion
                FROM departamento
              """
resultado = dd.sql(consultaSQL).df()

res = departamento[['id', 'descripcion']]

set(res) == set(resultado)

#%%-----------
# d. Listar todas las columnas de la tabla departamento.

consultaSQL = """
                SELECT *
                FROM departamento
              """
resultado = dd.sql(consultaSQL).df()

res = departamento
set(res) == set(resultado)

#%%-----------
# e. Listar los códigos de departamento y nombres de todos los departamentos que hay en la tabla departamento. Utilizar los siguientes alias para las columnas: codigo_depto y nombre_depto, respectivamente.

consultaSQL = """
                SELECT id AS codigo_depto, descripcion AS nombre_depto
                FROM departamento
              """
resultado = dd.sql(consultaSQL).df()

res = departamento[['id', 'descripcion']]
res = res.rename(columns={'id': 'codigo_depto', 'descripcion': 'nombre_depto'})

set(res) == set(resultado)

#%%-----------
# f. Listar los registros de la tabla departamento cuyo código de provincia es igual a 54

consultaSQL = """
                SELECT *
                FROM departamento
                WHERE id_provincia = 54
              """
resultado = dd.sql(consultaSQL).df()

res = departamento['id_provincia'] == 54
res = departamento[res]

set(res) == set(resultado)
set(res) == set(departamento[departamento['id_provincia'] == 54])

#%%-----------
# g. Listar los registros de la tabla departamento cuyo código de provincia es igual a 22, 78 u 86.

consultaSQL = """
                SELECT *
                FROM departamento
                WHERE id_provincia = 22 OR id_provincia = 78 OR id_provincia = 86
              """
resultado = dd.sql(consultaSQL).df()

res = departamento
res = (departamento['id_provincia'] == 22) | (departamento['id_provincia'] == 78) | (departamento['id_provincia'] == 86)
res = departamento[res]

set(res) == set(resultado)

#%%-----------
# h. Listar los registros de la tabla departamento cuyos códigos de provincia se encuentren entre el 50 y el 59 (ambos valores inclusive).

consultaSQL = """
                SELECT *
                FROM departamento
                WHERE id_provincia >= 50 AND id_provincia <= 59
                ORDER BY id_provincia ASC
              """
resultado = dd.sql(consultaSQL).df()

res = departamento
res = (departamento['id_provincia'] >= 50) & (departamento['id_provincia'] <= 59)
res = departamento[res]
res = res.sort_values('id_provincia')

set(res) == set(resultado)

#%%===========================================================================
# B. Consultas multitabla (INNER JOIN)
#%%===========================================================================
# a. Devolver una lista con los código y nombres de departamentos, asociados al nombre de la provincia al que pertenecen.

consultaSQL = """
                SELECT DISTINCT dep.id, dep.descripcion, prov.descripcion AS provincia
                FROM departamento AS dep
                INNER JOIN provincia as prov
                ON dep.id_provincia = prov.id
              """
resultado = dd.sql(consultaSQL).df()

dep = departamento.rename(columns={'descripcion': 'departamento'})
prov = provincia.rename(columns={'descripcion': 'provincia', 'id': 'id_provincia'})
res = dep.merge(prov, how='inner', on='id_provincia')
res = res[['id', 'departamento', 'provincia']]

#%%-----------
# b. Devolver una lista con los código y nombres de departamentos, asociados al nombre de la provincia al que pertenecen.

consultaSQL = """
                SELECT DISTINCT dep.id, dep.descripcion, prov.descripcion AS provincia
                FROM departamento AS dep
                INNER JOIN provincia as prov
                ON dep.id_provincia = prov.id
              """
resultado = dd.sql(consultaSQL).df()

#%%-----------
# c. Devolver los casos registrados en la provincia de “Chaco”.

consultaSQL = """
                SELECT count(*) AS 'Casos en Chaco'
                FROM casos
                INNER JOIN departamento as dep
                ON casos.id_depto = dep.id
                INNER JOIN provincia as prov
                ON dep.id_provincia = prov.id
                WHERE prov.descripcion = 'Chaco'
              """
resultado = dd.sql(consultaSQL).df()

dep = departamento.rename(columns = {'descripcion': 'departamento', 'id': 'id_depto'})
prov = provincia.rename(columns = {'id': 'id_provincia', 'descripcion': 'provincia'})

res = casos.merge(dep, how='inner', on='id_depto').merge(prov, how='inner', on='id_provincia')
res = res[['id_provincia', 'provincia']][res['provincia'] == 'Chaco']
res = pd.DataFrame(data = {'Casos en Chaco': [len(res)]})

#%%-----------
# d. Devolver aquellos casos de la provincia de “Buenos Aires” cuyo campo cantidad supere los 10 casos.

res = dd.sql("""
                SELECT *
                FROM casos
                INNER JOIN departamento AS dep
                ON casos.id_depto = dep.id
                WHERE casos.cantidad > 10
              """).df()

prov_bsas = dd.sql("""
                   SELECT id
                   FROM provincia
                   WHERE descripcion = 'Buenos Aires'
                   """).df()
consultaSQL = """
                SELECT *
                FROM res
                INNER JOIN provincia
                ON res.id_provincia = provincia.id
                WHERE (
                    SELECT p.id
                    FROM prov_bsas as p
                    WHERE provincia.id = p.id
                    )
              """
resultado = dd.sql(consultaSQL).df()

res = casos.merge(departamento.rename(columns = {'id': 'id_depto', 'descripcion': 'departamento'}), how='inner', on='id_depto')
res = res.merge(provincia.rename(columns = {'id': 'id_provincia', 'descripcion': 'provincia'}), how='inner', on='id_provincia')
res = res[(res['cantidad'] > 10) & (res['provincia'] == 'Buenos Aires')]

#%%===========================================================================
# C. Consultas multitabla (OUTER JOIN)
#%%===========================================================================
# a. Devolver un listado con los nombres de los departamentos que no tienen ningún caso asociado.

consultaSQL = """
                SELECT d.descripcion
                FROM departamento AS d
                LEFT JOIN casos AS c
                ON c.id_depto = d.id
                WHERE c.cantidad IS NULL
              """
resultado = dd.sql(consultaSQL).df()

res = departamento.rename(columns = {'id': 'id_depto', 'descripcion': 'departamento'}).merge(casos, how='left', on='id_depto')
res = res[res['cantidad'].isna()]['departamento']

#%%-----------
# b. Devolver un listado con los tipos de evento que no tienen ningún caso asociado.

consultaSQL = """
                SELECT *
                FROM casos
                LEFT JOIN tipoevento
                ON casos.id_tipoevento = tipoevento.id
                WHERE tipoevento.id IS NULL
              """
resultado = dd.sql(consultaSQL).df()

#%%===========================================================================
# D. Consultas resumen
#%%===========================================================================
#%% a. Calcular la cantidad total de casos que hay en la tabla casos.

consultaSQL = """
                SELECT SUM(cantidad) AS Cantidad
                FROM casos
              """
resultado = dd.sql(consultaSQL).df()

res = [int(casos['cantidad'].agg('sum'))]
res = pd.DataFrame(np.array(res), columns=['Cantidad'])

#%% b ------------------------------------------------------------------------

resultado = dd.sql("""
      SELECT e.descripcion, c.anio, c.cantidad
      FROM casos AS c, tipoevento AS e
      WHERE c.id_tipoevento = e.id
    """).df()
consultaSQL = """
                SELECT descripcion, anio, SUM(cantidad) AS Cantidad
                FROM resultado
                GROUP BY anio, descripcion
                ORDER BY anio ASC, descripcion ASC
              """
resultado = dd.sql(consultaSQL).df()

res = casos.merge(tipoevento.rename(columns = {'id': 'id_tipoevento', 'descripcion': 'tipo_caso'}), how='inner', on='id_tipoevento')
res = res[['anio', 'tipo_caso', 'cantidad']].groupby(['tipo_caso', 'anio'])['cantidad'].sum().reset_index()

#%% c ------------------------------------------------------------------------

consultaSQL = """
                SELECT descripcion AS tipo_caso, anio, cantidad
                FROM resultado
                WHERE anio = '2019'
              """
resultado = dd.sql(consultaSQL).df()

res = res[res['anio'] == 2019]
res = res.rename(columns = {'descripcion': 'tipo_caso'})

#%% d ------------------------------------------------------------------------

resultado = dd.sql("""
             SELECT prov.id AS codigo_prov, dep.descripcion AS Departamento, prov.descripcion AS Provincia
             FROM departamento AS dep, provincia AS prov
             WHERE dep.id_provincia = prov.id
             ORDER BY prov.id
             """).df()
1   
consultaSQL = """
                SELECT res.codigo_prov, res.provincia, COUNT(*) AS Cantidad
                FROM resultado AS res
                GROUP BY res.Provincia, res.codigo_prov
                ORDER BY res.codigo_prov ASC
              """
resultado = dd.sql(consultaSQL).df()

res = provincia.rename(columns = {'id': 'id_provincia', 'descripcion': 'provincia'})
res = res.merge(departamento.rename(columns = {'descripcion': 'cantidad_deptos'}), how='inner', on='id_provincia')
res = res[['id_provincia', 'provincia', 'cantidad_deptos']].groupby(['id_provincia', 'provincia'])['cantidad_deptos'].count().reset_index()

#%% e ------------------------------------------------------------------------

resultado = dd.sql("""
                   SELECT dep.descripcion AS departamento, MIN(c.cantidad) AS 'casos 2019'
                   FROM casos AS c
                   INNER JOIN departamento AS dep
                   ON c.id_depto = dep.id
                   GROUP BY dep.descripcion, c.anio
                   HAVING c.anio = 2019
                   """).df()

res = casos.merge(departamento.rename(columns = {'descripcion': 'departamento', 'id': 'id_depto'}), how='inner', on='id_depto')
res = res[['id_depto', 'departamento', 'cantidad', 'anio']].groupby(['departamento', 'anio'])['cantidad'].min().reset_index()
res = res[res['anio'] == 2019]
res = res.reset_index()
res = res[['departamento', 'cantidad']]

#%% f ------------------------------------------------------------------------

resultado = dd.sql("""
                   SELECT dep.descripcion AS departamento, MAX(c.cantidad) AS 'casos 2020'
                   FROM casos AS c
                   INNER JOIN departamento AS dep
                   ON c.id_depto = dep.id
                   GROUP BY dep.descripcion, c.anio
                   HAVING c.anio = 2020
                   ORDER BY "casos 2020" DESC
                   """).df()

res = casos.merge(departamento.rename(columns = {'descripcion': 'departamento', 'id': 'id_depto'}), how='inner', on='id_depto')
res = res[['departamento', 'cantidad', 'anio']].groupby(['departamento', 'anio'])['cantidad'].max().reset_index().sort_values(by='cantidad', ascending=False)
res = res[res['anio'] == 2020]
res = res.reset_index()
res = res[['departamento', 'cantidad']]

#%% g ------------------------------------------------------------------------

resultado = dd.sql("""
                   SELECT casos.id, casos.anio, casos.cantidad, dep.id_provincia
                   FROM casos, departamento AS dep
                   WHERE casos.id_depto = dep.id
                   """).df()
                   
resultado = dd.sql("""
                   SELECT  prov.descripcion AS provincia, res.anio, AVG(res.cantidad) AS promedio
                   FROM resultado AS res, provincia AS prov
                   WHERE res.id_provincia = prov.id
                   GROUP BY res.anio, provincia
                   ORDER BY res.anio ASC, provincia ASC
                   """).df()

res = departamento.rename(columns = {'id': 'id_depto', 'descripcion': 'departamento'})
res = casos.merge(res, how='inner', on='id_depto')
res = res.merge(provincia.rename(columns = {'id': 'id_provincia', 'descripcion': 'provincia'}), how='inner', on='id_provincia')
res = res[['provincia', 'anio', 'cantidad']].groupby(['provincia', 'anio'])['cantidad'].mean().reset_index().sort_values(by = 'anio')
res = res.reset_index()
res = res [['provincia', 'cantidad']]

#%% h ------------------------------------------------------------------------

resultado = dd.sql("""
                   SELECT c.anio, prov.descripcion AS provincia, depto.descripcion AS departamento, MAX(c.cantidad) AS maximo
                   FROM casos AS c, departamento AS depto, provincia as prov
                   WHERE c.id_depto = depto.id AND depto.id_provincia = prov.id
                   GROUP BY provincia, departamento, c.anio
                   ORDER BY c.anio, prov.descripcion, maximo
                    """).df()

#%% i ------------------------------------------------------------------------

resultado = dd.sql("""
                   SELECT COUNT(c.cantidad) AS total, MEAN(c.cantidad) AS promedio, MIN(c.cantidad) AS minimo, MAX(c.cantidad) AS maximo
                   FROM casos AS c, departamento AS depto, provincia AS prov
                   WHERE c.id_depto = depto.id AND depto.id_provincia = prov.id
                   GROUP BY prov.descripcion, c.anio
                   HAVING prov.descripcion = 'Buenos Aires' AND c.anio = 2019
                   """).df()

#%%===========================================================================
# E. Subconsultas (ALL, ANY)
#%%===========================================================================
#%% a ------------------------------------------------------------------------

resultado = dd.sql("""
                   SELECT dep.descripcion, c1.cantidad
                   FROM casos AS c1, departamento AS dep
                   WHERE c1.id_depto = dep.id AND c1.cantidad >= ALL (
                       SELECT c2.cantidad
                       FROM casos AS c2
                       WHERE c1.id_depto = c2.id_depto
                       )
                   """).df()
                   
#%% b ------------------------------------------------------------------------

resultado = dd.sql("""
                   SELECT t1.descripcion AS evento
                   FROM casos AS c1, tipoevento AS t1
                   WHERE c1.id_tipoevento = t1.id AND t1.id = ANY (
                       SELECT c2.id
                       FROM casos AS c2
                       WHERE c1.id = c2.id
                       )
                   """).df()

#%%===========================================================================
# F. Subconsultas (IN, NOT IN)
#%%===========================================================================
#%% a ------------------------------------------------------------------------

resultado = dd.sql("""
                   SELECT t1.descripcion AS tipo_evento
                   FROM tipoevento AS t1
                   WHERE t1.id IN (
                       SELECT c.id_tipoevento
                       FROM casos AS c
                       )
                   """).df()

#%% b ------------------------------------------------------------------------

resultado = dd.sql("""
                   SELECT t1.descripcion AS tipo_evento
                   FROM tipoevento AS t1
                   WHERE t1.id NOT IN (
                       SELECT c.id_tipoevento
                       FROM casos AS c
                       )
                   """).df()
                   
#%%===========================================================================
# G. Subconsultas (EXISTS, NOT EXISTS)
#%%===========================================================================
#%% a ------------------------------------------------------------------------

resultado = dd.sql("""
                   SELECT t.descripcion AS tipo_evento
                   FROM tipoevento AS t
                   WHERE EXISTS (
                       SELECT c.id_tipoevento
                       FROM casos AS c
                       WHERE c.id_tipoevento = t.id
                       )
                   """).df()
                   
#%% b ------------------------------------------------------------------------

resultado = dd.sql("""
                   SELECT t.descripcion AS tipo_evento
                   FROM tipoevento AS t
                   WHERE NOT EXISTS (
                       SELECT c.id_tipoevento
                       FROM casos AS c
                       WHERE c.id_tipoevento = t.id
                       )
             """).df()
                   
#%%===========================================================================
# H. Subconsultas correlacionadas
#%%===========================================================================
#%% a ------------------------------------------------------------------------

print(casos['cantidad'].agg('mean'))
resultado = dd.sql("""
                   SELECT p.descripcion AS provincias, c1.cantidad
                   FROM provincia AS p, casos AS c1, departamento AS d
                   WHERE (c1.id_depto = d.id) AND (p.id = d.id_provincia) AND c1.cantidad > (
                       SELECT MEAN(c2.cantidad)
                       FROM casos AS c2
                       )
                   """).df()

#%% b ------------------------------------------------------------------------

resultado = dd.sql("""
                   SELECT c1.anio, p1.descripcion AS provincia, COUNT(c1.cantidad) AS cantidad
                   FROM casos AS c1, provincia AS p1, departamento AS d1
                   WHERE (c1.id_depto = d1.id) AND (d1.id_provincia = p1.id)
                   GROUP BY c1.anio, p1.descripcion
                   HAVING cantidad > (
                       SELECT COUNT(c2.cantidad)
                       FROM casos AS c2, provincia AS p2, departamento Ad2
                       WHERE (c2.id_depto = d2.id) AND (d2.id_provincia = p2.id)
                       GROUP BY c2.anio, p2.descripcion
                       HAVING p2.descripcion = 'Corrientes'
                       )
                   """).df()

#%%===========================================================================
# I. Más consultas sobre una tabla
#%%===========================================================================
#%% a ------------------------------------------------------------------------

resultado = dd.sql("""
                  SELECT id AS codigo, descripcion AS nombre
                  FROM departamento
                  ORDER BY nombre DESC, codigo ASC
                  """).df()

#%% b ------------------------------------------------------------------------

resultado = dd.sql("""
                   SELECT *
                   FROM provincia
                   WHERE descripcion LIKE 'M%'
                   """).df()

#%% c ------------------------------------------------------------------------

resultado = dd.sql("""
                   SELECT *
                   FROM provincia
                   WHERE descripcion LIKE 'S___a%'
                   """).df()

#%% d ------------------------------------------------------------------------

resultado = dd.sql("""
                   SELECT *
                   FROM provincia
                   GROUP BY descripcion, id
                   HAVING descripcion LIKE '%a'
                   """).df()
        
resultado = dd.sql("""
                   SELECT *
                   FROM provincia
                   WHERE descripcion LIKE '%a'
                   """).df()   

#%% e ------------------------------------------------------------------------

resultado = dd.sql("""
                   SELECT *
                   FROM provincia
                   WHERE descripcion LIKE '_____'
                   """).df()

#%% f ------------------------------------------------------------------------
resultado = dd.sql("""
                   SELECT *
                   FROM provincia
                   WHERE descripcion LIKE '%do%'
                   """).df()

#%% g ------------------------------------------------------------------------

resultado = dd.sql("""
                   SELECT *
                   FROM provincia
                   WHERE descripcion LIKE '%do%' AND id < 30
                   """).df()

#%% h ------------------------------------------------------------------------

resultado = dd.sql("""
                   SELECT id AS codigo_depto, descripcion AS nombre_depto
                   FROM departamento
                   WHERE nombre_depto LIKE '%san%' OR nombre_depto LIKE '%San%'
                   ORDER BY nombre_depto DESC
                   """).df()

#%% i ------------------------------------------------------------------------

resultado = dd.sql("""
                   SELECT p.descripcion AS provincia, d.descripcion AS departamento, c.anio, c.semana_epidemiologica AS semana, e.descripcion AS edad, c.cantidad AS cantidad
                   FROM casos AS c, provincia AS p, departamento AS d, grupoetario AS e
                   WHERE (c.id_depto = d.id) AND (d.id_provincia = p.id) AND (c.id_grupoetario = e.id) AND (provincia LIKE '%a')
                   ORDER BY cantidad DESC, provincia ASC, departamento ASC, anio ASC, edad ASC
                   """).df()

#%% j ------------------------------------------------------------------------

resultado = dd.sql("""
                   SELECT provincia, departamento, anio, semana, edad, MAX(cantidad) AS cantidad
                   FROM resultado
                   GROUP BY provincia, departamento, anio, semana, edad, cantidad
                   """).df()

#%%===========================================================================
# J. Reemplazos
#%%===========================================================================
#%% a ------------------------------------------------------------------------

resultado = dd.sql("""
                  SELECT id, REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(descripcion,  'á', 'a'), 'é', 'e'), 'í', 'i'), 'Ó', 'O'), 'Ú', 'U') AS descripcion
                  FROM departamento
                  ORDER BY descripcion ASC
                  """).df()

#%% b ------------------------------------------------------------------------

resultado = dd.sql("""
                   SELECT id, UPPER(descripcion) AS descripcion
                   FROM resultado
                   """).df()


