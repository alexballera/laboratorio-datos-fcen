# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 10:11:23 2025

@author: ICBC
"""
#%%-----------
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import duckdb as dd
import seaborn as sns
import matplotlib.ticker as ticker

#%%-----------
wine = pd.read_csv('wine.csv', sep=';')

plt.scatter(data = wine, x='fixed acidity', y='citric acid')

#%%-----------
fig, ax = plt.subplots()

# plt.rcParams['font_family'] = 'sans-serif'
ax.scatter(data = wine,
           x='fixed acidity',
           y='citric acid',
           s=2, # Tamaño de los puntos
           color='magenta' # color de los puntos
           )

ax.set_title('Acidez vs contenido de ácido cítrico')

#%%-----------
arboles = pd.read_csv('arbolado-en-espacios-verdes.csv', index_col=2)

mas_frecuentes = arboles['nombre_com'].value_counts()[:30]

mas_frec_index = mas_frecuentes.reset_index()

consultaSQL = """
        SELECT *
        FROM arboles AS a
        WHERE a.nombre_com IN (
            SELECT f.nombre_com
            FROM mas_frec_index AS f
            )
      """
res = dd.sql(consultaSQL).df()

#%%-----------
# Ejercicio 1
fig, ax = plt.subplots()
ax.scatter(data = res,
            x='diametro',
            y='altura_tot',
            s=1, # Tamaño de los puntos
            color='magenta' # color de los puntos
            )
ax.set_title('Diámetro vs altura')
#%%-----------
# Ejercicio 2
fig, ax = plt.subplots()
ax.scatter(data = res,
            x='long',
            y='lat',
            s=1, # Tamaño de los puntos
            color='cyan' # color de los puntos
            )
ax.set_title('Long vs Lat')

#%%-----------
# Ejercicio 3
origens = ['Exótico', 'Nativo/Autóctono', 'No Determinado']
colors = {'No Determinado': 'orange', 'Nativo/Autóctono': 'green', 'Exótico': 'cyan'}

#%%-----------

fig, ax = plt.subplots()
plt.rcParams['font.family'] = 'sans-serif'

tamanoBurbuja = 5

ax.scatter(data = wine,
           x='fixed acidity',
           y='citric acid',
           s=wine['residual sugar']*tamanoBurbuja # Se agrega una variable adicional
           )

ax.set_title('Relación entre tres variables')
ax.set_xlabel('Acidez (g/dm3)', fontsize='medium')
ax.set_ylabel('Contenido de ácido cítrico (g/dm3)', fontsize='medium')

#%%-----------
fig, ax = plt.subplots()
wine['type'].value_counts().plot(
    kind = 'pie',
    ax = ax,
    autopct = '%1.1f%%',
    colors=['#66b3ff','#ff9999'],
    startangle=90,
    shadow=True,
    explode=(0.1, 0),
    legend=False

    )
ax.set_ylabel('')
ax.set_title('Distribución de Tipos de Vino')
#%%-----------
fig, ax = plt.subplots()
wine = pd.read_csv('wine.csv', sep=';')

ax.scatter(data = wine, x='pH', y='chlorides')
ax.set_title('Relación pH de Vino')
ax.set_xlabel('pH', fontsize='medium')
ax.set_ylabel('Chlorides', fontsize='medium')
fig.savefig('relacion_ph_vinos')

#%%-----------
fig, ax = plt.subplots()
plt.rcParams['font.family'] = 'sans-serif'
cheetahRegion = pd.read_csv('cheetahRegion.csv')

ax.bar(data = cheetahRegion, x='Anio', height='Ventas')
ax.set_xlabel('Año', fontsize='medium')
ax.set_ylabel('Ventas - MM $', fontsize='medium')
ax.set_xlim(0, 11)
ax.set_ylim(0, 250)

ax.set_xticks(range(1,11,1))
ax.set_yticks([])
ax.bar_label(ax.containers[0], fontsize=8)

#%%-----------
fig, ax = plt.subplots()
plt.rcParams['font.family'] = 'sans-serif'
cheetahRegion = pd.read_csv('cheetahRegion.csv')

cheetahRegion.plot(x='Anio',
                   y=['regionEste', 'regionOeste'],
                   kind='bar',
                   label=['Región Este', 'Region Oeste'],
                   ax = ax
                   )

#%%-----------
# 7 Gráficos de líneas
fig, ax = plt.subplots()

plt.rcParams['font.family'] = 'sans-serif'

ax.plot('Anio', 'Ventas', data=cheetahRegion, marker='o')

ax.set_xlabel('Año', fontsize='medium')
ax.set_ylabel('Ventas - MM $', fontsize='medium')
ax.set_xlim(0, 12)
ax.set_ylim(0, 250)
ax.set_title('Ventas de la compañia')

#%%-----------
# 8 Gráficos de líneas
fig, ax = plt.subplots()

plt.rcParams['font.family'] = 'sans-serif'

# regionEste
ax.plot(
        'Anio',
        'regionEste',
        data=cheetahRegion,
        marker='o',
        )



#%%-----------
# Distribución Datos Categóricos

fig, ax = plt.subplots()
gaseosas = pd.read_csv('gaseosas.csv')

gaseosas['Compras_gaseosas'].value_counts().plot.bar(ax = ax)

#%%-----------
# Distribución Datos Continuos
ageAtDeath = pd.read_csv('ageAtDeath.csv')

sns.histplot(data = ageAtDeath['AgeAtDeath'], bins = 20)

#%%-----------
# Distribución Datos Continuos
tips = pd.read_csv('tips.csv')
sns.histplot(data = tips, x ='tip', bins = 40, hue='sex')
#%%-----------
# Medidas central y dispersiòn
tips = pd.read_csv('tips.csv')
tips['tip'].mean()
tips['tip'].median()
tips['tip'].describe()
#%%-----------
# Boxplot
ventaCasas = pd.read_csv('ventaCasas.csv')

ventaCasas['PrecioDeVenta'].describe()
ventaCasas['PrecioDeVenta'].median()

fig, ax = plt.subplots()

ax.boxplot(ventaCasas['PrecioDeVenta'], showmeans = True)

ax.set_title('Precio de venta casas')
ax.set_xticks([])
ax.set_ylabel('Precio de venta ($)')
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("$ {x:,.2f}"))
ax.set_ylim(0,500)
#%%-----------
# Boxplot
fig, ax = plt.subplots()

tips.boxplot(
    by=['sex'],
    column=['tip'],
    ax=ax,
    grid=False,
    showmeans=True
    )

fig.suptitle('')
ax.set_title('Propinas')
ax.set_xlabel('Sexo')
ax.set_ylabel('Valor de la propina')
#%%-----------
# Boxplot
tips = pd.read_csv('tips.csv')

ax = sns.violinplot(
    x='sex',
    y='tip',
    data=tips,
    palette={
        'Female': 'orange',
        'Male': 'skyblue'
        }
    )

ax.set_title('Propinas')
ax.set_xlabel('sexo')
ax.set_ylabel('Valor de propina')
ax.set_ylim(0, 12)
ax.set_xticklabels(['Femenino', 'Masculino'])
#%%-----------
# Boxplot
ventaCasas = pd.read_csv('ventaCasas.csv')

ventaCasas['PrecioDeVenta'].describe()
ventaCasas['PrecioDeVenta'].median()

ax = sns.violinplot(y='PrecioDeVenta', data=ventaCasas, fill=False)

ax.set_title('Precio de venta casas')
ax.set_xticks([])
ax.set_ylabel('Precio de venta ($)')
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("$ {x:,.2f}"))
ax.set_ylim(-50,600)
#%%-----------
