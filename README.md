# 📊 Laboratorio de Datos - FCEn UBA

Bienvenido/a al repositorio oficial del curso **Laboratorio de Datos** de la **Tecnicatura en Ciencia de Datos** de la Facultad de Ciencias Exactas y Naturales (FCEn) de la Universidad de Buenos Aires (UBA).

Este repositorio contiene todo el material didáctico del curso: notebooks interactivos, scripts de ejemplo, datasets para prácticas, ejercicios y trabajos prácticos evaluables.

---

## 📋 Sobre el Curso

### Objetivos del curso

Aprender a analizar datos utilizando Python como herramienta principal, cubriendo:

- **Manipulación de datos** con Pandas y NumPy
- **Bases de datos relacionales** y SQL
- **Calidad de datos** y limpieza
- **Visualización** y análisis exploratorio
- **Machine Learning** supervisado y no supervisado
- **Metodología** de análisis de datos

### Audiencia

Estudiantes de economía, ciencias sociales y disciplinas afines que desean aprender análisis de datos desde cero. **No se requiere experiencia previa en programación**.

---

## 🚀 Inicio Rápido

### Prerrequisitos

- **Python 3.8 o superior** (recomendado: Python 3.12+)
- **Git** para clonar el repositorio
- **Editor de código** o Jupyter Notebook/Lab

### Instalación en 4 pasos

#### 1. Clonar el repositorio

```bash
git clone https://github.com/alexballera/laboratorio-fcen.git
cd laboratorio-fcen
```

#### 2. Crear entorno virtual

**Linux/macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

#### 3. Instalar dependencias

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Verificar instalación

```bash
python test_environment.py
```

Si ves ✅ en todos los componentes, ¡estás listo/a para comenzar!

---

## 📚 Contenido del Curso

El curso está organizado en **5 módulos progresivos**, cada uno con clases teóricas y prácticas:

### Módulo 1: Fundamentos de Python y Pandas (Clases 0-1)

**¿Qué aprenderás?**
- Sintaxis básica de Python: variables, tipos, estructuras de control
- Manipulación de archivos de texto y CSV
- Pandas: creación, indexación, filtrado y transformación de DataFrames
- NumPy: arrays y operaciones vectorizadas

**Carpetas:**
- [`Clase-00-PresentacionMateria/`](clases/Clase-00-PresentacionMateria/)
- [`Clase-01-PythonPandas/`](clases/Clase-01-PythonPandas/)

**Ejercicios clave:**
- Análisis del arbolado público de CABA
- Procesamiento de datos de movilidad

---

### Módulo 2: Bases de Datos y SQL (Clases 2-9)

**¿Qué aprenderás?**
- Metodología de análisis de datos
- Diseño de bases de datos: Diagramas Entidad-Relación (DER)
- Modelo relacional y claves (primarias, foráneas)
- Normalización: 1FN, 2FN, 3FN, BCNF
- SQL con DuckDB: SELECT, JOIN, GROUP BY, subconsultas
- Álgebra relacional

**Carpetas:**
- [`Clase-02-introMetodología/`](clases/Clase-02-introMetodología/)
- [`Clase-03-ModeladoDeDatos-DER/`](clases/Clase-03-ModeladoDeDatos-DER/)
- [`Clase-04-ModeloRelacional/`](clases/Clase-04-ModeloRelacional/)
- [`Clase-05-06-07-AlgebraRelacional-SQL/`](clases/Clase-05-06-07-AlgebraRelacional-SQL/)
- [`Clase-08--09-Normalizacion/`](clases/Clase-08--09-Normalizacion/)

**Datasets:**
- Encuesta de movilidad
- Datos procesados de SQL

---

### Módulo 3: Calidad de Datos y Visualización (Clases 10-12)

**¿Qué aprenderás?**
- Detección y corrección de errores en datos
- Imputación de valores faltantes
- Análisis Exploratorio de Datos (AED)
- Matplotlib y Seaborn: histogramas, scatter plots, boxplots, heatmaps
- Principios de diseño de visualizaciones efectivas

**Carpetas:**
- [`Clase-10-CalidadDeDatos/`](clases/Clase-10-CalidadDeDatos/)
- [`Clase-11-12-Visualizacion AED/`](clases/Clase-11-12-Visualizacion%20AED/)

**Datasets:**
- Datos de dengue y zika
- Aeropuertos, gaseosas, vinos, tips, etc.

---

### Módulo 4: Machine Learning Supervisado (Clases 13-18)

**¿Qué aprenderás?**
- Introducción al modelado predictivo
- Clasificación: Árboles de Decisión, Random Forest, K-Nearest Neighbors (KNN)
- Métricas de evaluación: accuracy, precision, recall, F1-score
- Regresión Lineal Simple (RLS)
- Selección de modelos: train/test split, cross-validation
- Overfitting y underfitting

**Carpetas:**
- [`Clase-13-IntroModelado/`](clases/Clase-13-IntroModelado/)
- [`Clase-14-15-Clasificacion/`](clases/Clase-14-15-Clasificacion/)
- [`clase-16- RLS/`](clases/clase-16-%20RLS/)
- [`Clase-17-RegresiónKNN/`](clases/Clase-17-RegresiónKNN/)
- [`Clase-18-SeleccionModelos/`](clases/Clase-18-SeleccionModelos/)

**Datasets de práctica:**
- Titanic (clasificación binaria)
- Iris (clasificación multiclase)
- Datos de tiempo de reacción

---

### Módulo 5: Machine Learning No Supervisado (Clase 19)

**¿Qué aprenderás?**
- Clustering: K-Means, DBSCAN
- Reducción de dimensionalidad: PCA (Principal Component Analysis)

**Carpetas:**
- [`Clase-19-NoSupervisado/`](clases/Clase-19-NoSupervisado/)

---

## 📂 Estructura del Repositorio

```
laboratorio-fcen/
│
├── README.md                    # 👈 Estás aquí
├── AGENTS.md                    # Instrucciones completas para IA
├── ENVIRONMENT_SETUP.md         # Guía detallada de instalación
├── requirements.txt             # Dependencias del curso
├── test_environment.py          # Script de verificación
│
├── clases/                      # 📚 Material por clase
│   ├── Clase-00-PresentacionMateria/
│   ├── Clase-01-PythonPandas/
│   │   ├── clase/              # Ejemplos de la clase teórica
│   │   └── practica/           # Ejercicios prácticos
│   ├── Clase-02-introMetodología/
│   ├── ...
│   └── evaluaciones/           # 📝 Trabajos prácticos (TPs)
│       ├── tp1/
│       ├── tp2/
│       └── parciales/
│
├── bibliografia/               # 📖 Material de lectura
├── cheatsheets/               # 📄 Guías rápidas de referencia
└── .github/
    └── skills/
        └── tutor-profesor/    # 🤖 Agente IA tutor del curso
```

---

## 🎯 Cómo Usar Este Repositorio

### Para estudiantes

1. **Navega por módulos**: Empieza por `Clase-01-PythonPandas/` si es tu primera vez
2. **Lee las consignas**: Cada carpeta `practica/` tiene ejercicios con instrucciones
3. **Ejecuta los notebooks**: Abre archivos `.ipynb` con Jupyter o VS Code
4. **Practica con datasets reales**: Usa los archivos CSV incluidos
5. **Consulta al tutor IA**: Escribe `/tutor-profesor` en GitHub Copilot para ayuda pedagógica ([ver guía completa](GUIA_TUTOR.md))

### Para docentes

1. **Reutiliza material**: Los notebooks están diseñados para ser modulares
2. **Adapta ejercicios**: Modifica datasets o consignas según tu audiencia
3. **Contribuye mejoras**: Envía PRs con correcciones o nuevos ejercicios
4. **Revisa trabajos**: Usa las rúbricas del agente tutor-profesor

---

## 💻 Ejecutar Notebooks y Scripts

### Opción 1: Jupyter Notebook (recomendado para principiantes)

```bash
source .venv/bin/activate  # Activar entorno
jupyter notebook           # Abrir navegador con interfaz
```

Navega a `clases/Clase-XX/` y abre los archivos `.ipynb`.

### Opción 2: Jupyter Lab (interfaz moderna)

```bash
source .venv/bin/activate
jupyter lab
```

### Opción 3: VS Code (para usuarios avanzados)

1. Abre la carpeta del repositorio en VS Code
2. Selecciona el intérprete de Python del entorno virtual (`.venv`)
3. Instala la extensión de Jupyter si no la tienes
4. Abre archivos `.ipynb` directamente en el editor

### Ejecutar scripts Python

```bash
source .venv/bin/activate
python clases/Clase-01-PythonPandas/practica/pandas_script1.py
```

---

## 🧪 Verificación y Tests

### Test básico de entorno

```bash
python test_environment.py
```

Verifica que todas las librerías principales estén instaladas y funcionales.

### Ejecutar pruebas adicionales (si existen)

```bash
pytest -q
```

---

## 🤖 Asistente Tutor-Profesor IA

Este repositorio incluye un **agente IA especializado** diseñado para actuar como tutor del curso.

### ¿Qué puede hacer?

- ✅ Revisar tus notebooks y darte feedback constructivo
- ✅ Explicar conceptos de Python, Pandas, SQL, Machine Learning
- ✅ Proponer ejercicios adaptados a tu nivel
- ✅ Corregir errores en tu código con explicaciones didácticas
- ✅ Guiarte en trabajos prácticos sin darte la solución directa
- ✅ Evaluar trabajos con criterios pedagógicos claros

### ¿Cómo usarlo?

Si usas **GitHub Copilot** en VS Code, hay dos formas:

#### Opción 1: Selector de Agentes (Recomendado)
1. Abre el chat de Copilot (Ctrl+I o Cmd+I)
2. Haz clic en el selector de agentes (menú desplegable arriba)
3. Selecciona **"Tutor Laboratorio Datos"**
4. Escribe tus consultas normalmente

```
¿Cómo hago un groupby en Pandas?
```

#### Opción 2: Comando Slash
Escribe `/tutor-profesor` seguido de tu consulta:

```
/tutor-profesor revisar mi notebook de la Clase 5
```

El agente responderá **siempre en español**, con un tono didáctico y ejemplos prácticos.

**📖 Guía completa**: Ver [GUIA_TUTOR.md](GUIA_TUTOR.md) para ejemplos detallados y mejores prácticas.

---

## 📖 Recursos Adicionales

### Documentación oficial de librerías

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [NumPy User Guide](https://numpy.org/doc/stable/user/)
- [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/index.html)
- [Seaborn Gallery](https://seaborn.pydata.org/examples/index.html)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [DuckDB Documentation](https://duckdb.org/docs/)

### Cheat sheets (guías rápidas)

Dentro de la carpeta [`cheatsheets/`](cheatsheets/) encontrarás resúmenes visuales de:
- Pandas
- NumPy
- Matplotlib
- SQL

### Bibliografía

Ver carpeta [`bibliografia/`](bibliografia/) para papers, libros y artículos recomendados.

---

## 🤝 Contribuciones

Este repositorio está abierto a contribuciones de estudiantes, docentes y colaboradores.

### ¿Cómo contribuir?

1. **Reportar errores**: Abre un [Issue](https://github.com/alexballera/laboratorio-fcen/issues)
2. **Sugerir mejoras**: Propone nuevos ejercicios o datasets
3. **Corregir código**: Envía un Pull Request con tus cambios

### Antes de enviar un PR

- [ ] Ejecutar `python test_environment.py` sin errores
- [ ] Verificar que notebooks se ejecutan completamente
- [ ] Actualizar `requirements.txt` si agregaste dependencias
- [ ] Documentar cambios en el mensaje del commit

---

## 📝 Trabajos Prácticos (TPs)

Los trabajos prácticos evaluables están en [`clases/evaluaciones/`](clases/evaluaciones/).

### Estructura típica de un TP

```
tp1/
├── consignas.md          # Enunciado del trabajo
├── datos/                # Datasets necesarios
├── plantilla.ipynb       # Notebook base para completar
└── solucion_ejemplo.ipynb # (Solo para docentes)
```

### Criterios de evaluación

- **Correctitud (40%)**: ¿Las respuestas son correctas? ¿El código funciona?
- **Metodología (30%)**: ¿Proceso de análisis lógico y justificado?
- **Claridad (20%)**: ¿Código legible y bien documentado?
- **Prolijidad (10%)**: ¿Estructura ordenada sin código redundante?

Para más detalles, consulta al agente tutor-profesor: `/tutor-profesor criterios de evaluación`.

---

## ⚙️ Solución de Problemas

### Error: "ModuleNotFoundError"

**Causa**: Dependencias no instaladas o entorno virtual no activado.

**Solución**:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Error: "Kernel not found" en Jupyter

**Causa**: El kernel no apunta al entorno virtual.

**Solución**:
```bash
python -m ipykernel install --user --name=labo-datos
# Luego selecciona "labo-datos" como kernel en Jupyter
```

### Error: "Permission denied" al activar entorno (Windows)

**Causa**: Política de ejecución de scripts de PowerShell.

**Solución**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 📧 Contacto y Soporte

- **Docente responsable**: Alexander Ballera
- **Repositorio**: [github.com/alexballera/laboratorio-fcen](https://github.com/alexballera/laboratorio-fcen)
- **Issues**: [Reportar un problema](https://github.com/alexballera/laboratorio-fcen/issues)

### Horarios de consulta

Consultar con el docente en clase o por los canales oficiales de la materia.

---

## 📜 Licencia

Este proyecto está bajo la licencia **MIT**. Ver archivo [`LICENSE`](LICENSE) para más detalles.

Esto significa que puedes:
- ✅ Usar el material con fines educativos
- ✅ Modificar y adaptar los ejercicios
- ✅ Compartir con atribución al autor original

---

## 🎓 Sobre la Facultad

**Facultad de Ciencias Exactas y Naturales - UBA**  
Tecnicatura en Ciencia de Datos

Este curso es parte del plan de estudios de la Tecnicatura en Ciencia de Datos, una carrera de pregrado orientada a formar profesionales en análisis y visualización de datos.

---

## 🌟 Agradecimientos

A todos los estudiantes que han contribuido con feedback y mejoras al material del curso.

---

**¡Bienvenido/a al mundo del análisis de datos! 🚀📊**
