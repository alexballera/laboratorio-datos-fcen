#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de prueba para verificar el entorno Python del proyecto
Laboratorio de Datos - FCEn UBA

Este script prueba las funcionalidades principales utilizadas
en el curso usando ejemplos similares a los del proyecto.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import duckdb as dd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def main():
    print("🔬 PRUEBA DE ENTORNO - LABORATORIO DE DATOS FCEn UBA")
    print("=" * 60)
    
    # 1. Prueba de Pandas y NumPy (Módulo 1)
    print("\n📊 1. Probando manipulación de datos con Pandas...")
    data = {
        'nombre': ['Ana', 'Bob', 'Carlos', 'Diana'],
        'edad': [25, 30, 35, 28],
        'salario': [50000, 60000, 70000, 55000]
    }
    df = pd.DataFrame(data)
    print(f"   ✓ DataFrame creado: {df.shape[0]} filas, {df.shape[1]} columnas")
    print(f"   ✓ Promedio de edad: {df['edad'].mean():.1f} años")
    
    # 2. Prueba de DuckDB (Módulo 2)
    print("\n🗃️  2. Probando consultas SQL con DuckDB...")
    query = """
        SELECT nombre, edad, salario
        FROM df
        WHERE edad > 27
        ORDER BY salario DESC
    """
    resultado = dd.sql(query).df()
    print(f"   ✓ Consulta SQL ejecutada: {len(resultado)} registros encontrados")
    
    # 3. Prueba de Visualización (Módulo 3)
    print("\n📈 3. Probando visualización con Matplotlib/Seaborn...")
    plt.figure(figsize=(8, 5))
    
    # Crear subplots
    plt.subplot(1, 2, 1)
    plt.scatter(df['edad'], df['salario'], c='blue', alpha=0.7)
    plt.xlabel('Edad')
    plt.ylabel('Salario')
    plt.title('Relación Edad-Salario')
    
    plt.subplot(1, 2, 2)
    sns.barplot(data=df, x='nombre', y='edad')
    plt.title('Edad por Persona')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('test_environment_plot.png', dpi=150, bbox_inches='tight')
    plt.close()  # No mostrar en terminal
    print("   ✓ Gráficos creados y guardados como 'test_environment_plot.png'")
    
    # 4. Prueba de Machine Learning (Módulo 4)
    print("\n🤖 4. Probando Machine Learning con Scikit-learn...")
    
    # Cargar dataset Iris (similar al usado en clase)
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    
    # División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Entrenar KNN (usado en el proyecto)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    acc_knn = accuracy_score(y_test, y_pred_knn)
    
    # Entrenar Árbol de Decisión (usado en el proyecto)
    tree = DecisionTreeClassifier(random_state=42)
    tree.fit(X_train, y_train)
    y_pred_tree = tree.predict(X_test)
    acc_tree = accuracy_score(y_test, y_pred_tree)
    
    print(f"   ✓ KNN (k=3) - Accuracy: {acc_knn:.3f}")
    print(f"   ✓ Decision Tree - Accuracy: {acc_tree:.3f}")
    print(f"   ✓ Dataset Iris: {X.shape[0]} muestras, {X.shape[1]} características")
    
    # 5. Prueba de análisis estadístico
    print("\n📊 5. Probando análisis estadístico...")
    
    # Estadísticas descriptivas (común en el curso)
    stats = X.describe()
    print(f"   ✓ Estadísticas calculadas para {len(X.columns)} variables")
    
    # Normalización (usado en clustering)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"   ✓ Datos normalizados: media ≈ {np.mean(X_scaled):.3f}, std ≈ {np.std(X_scaled):.3f}")
    
    print("\n" + "=" * 60)
    print("🎉 ¡ENTORNO CONFIGURADO CORRECTAMENTE!")
    print("📚 Todas las librerías del curso están funcionando.")
    print("🚀 ¡Listo para trabajar con el Laboratorio de Datos!")
    print("=" * 60)

if __name__ == "__main__":
    main()