# Configuración del Entorno Python - Laboratorio de Datos FCEn UBA

## 🐍 Entorno Python Configurado

- **Python Version**: 3.12.3
- **Environment Type**: Virtual Environment (.venv)
- **Status**: ✅ Configurado y funcionando

## 📦 Librerías Instaladas

### Core Data Science
- **NumPy**: 2.3.5 - Computación numérica
- **Pandas**: 2.3.3 - Manipulación de datos
- **SciPy**: 1.16.3 - Algoritmos científicos

### Machine Learning
- **Scikit-learn**: 1.7.2 - Algoritmos de ML y métricas

### Visualización
- **Matplotlib**: 3.10.7 - Gráficos y visualizaciones
- **Seaborn**: 0.13.2 - Gráficos estadísticos

### Base de Datos
- **DuckDB**: 1.4.2 - SQL analytics en Python

### Desarrollo
- **Jupyter**: 1.1.1 - Notebooks interactivos
- **IPython**: 9.7.0 - Python interactivo

## 🚀 Comandos de Activación

### Linux/Mac:
```bash
source .venv/bin/activate
```

### Windows:
```cmd
.venv\Scripts\activate
```

## 📋 Scripts de Verificación

### Verificación Rápida:
```bash
python test_environment.py
```

### Verificación Manual:
```bash
python -c "import pandas, numpy, sklearn, matplotlib, seaborn, duckdb; print('✅ Entorno OK')"
```

## 📚 Uso por Módulo del Curso

### Módulo 1 - Python/Pandas:
```python
import pandas as pd
import numpy as np
```

### Módulo 2 - SQL/Bases de Datos:
```python
import pandas as pd
import duckdb as dd
```

### Módulo 3 - Visualización:
```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
```

### Módulo 4 - Machine Learning:
```python
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
```

## 🔧 Instalación Desde Cero

```bash
# 1. Crear entorno virtual
python -m venv .venv

# 2. Activar entorno
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Verificar instalación
python test_environment.py
```

## 📁 Archivos de Configuración

- `requirements.txt` - Lista de dependencias
- `test_environment.py` - Script de verificación
- `.venv/` - Entorno virtual (no incluir en Git)

## ⚠️ Notas Importantes

- El entorno virtual está configurado localmente
- Ejecutar siempre desde el directorio del proyecto
- Activar el entorno antes de trabajar
- El archivo `.venv/` está excluido del repositorio Git

## 🎯 Listo para el Curso

¡El entorno está completamente configurado para trabajar con todos los módulos del Laboratorio de Datos!