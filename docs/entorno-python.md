# Entorno Python: configuración y gestión de dependencias

Guía de referencia para crear entornos virtuales Python y administrar dependencias con `pip-tools` en este proyecto.

---

## Conceptos clave

| Concepto | Descripción |
|---|---|
| **Entorno virtual** | Carpeta aislada con su propia instalación de Python y paquetes, separada del sistema global. |
| `requirements.in` | Archivo de dependencias **directas** mantenido manualmente. La "fuente de verdad" del proyecto. |
| `requirements.txt` | Archivo **generado automáticamente** por `pip-tools`. Contiene todas las dependencias con versiones exactas (pinned). No editar a mano. |
| `pip-tools` | Herramienta que resuelve el árbol de dependencias y genera `requirements.txt` a partir de `requirements.in`. |

---

## 1. Crear el entorno virtual

```bash
python -m venv .venv
```

Esto crea una carpeta `.venv/` en el directorio del proyecto con una instalación aislada de Python.

> En el control de versiones (Git), la carpeta `.venv/` está ignorada por `.gitignore`. No se sube al repositorio.

---

## 2. Activar el entorno virtual

El entorno debe estar activo en cada sesión de trabajo antes de instalar paquetes o ejecutar notebooks.

| Sistema operativo | Comando |
|---|---|
| Windows (CMD) | `.venv\Scripts\activate.bat` |
| Windows (PowerShell) | `.venv\Scripts\Activate.ps1` |
| Windows (Git Bash) | `source .venv/Scripts/activate` |
| Linux / macOS | `source .venv/bin/activate` |

Una vez activo, el prompt del terminal muestra `(.venv)` al inicio.

Para **desactivar** el entorno:

```bash
deactivate
```

---

## 3. Instalar pip-tools

Con el entorno activo, instalar `pip-tools` una sola vez:

```bash
pip install pip-tools
```

---

## 4. Instalar las dependencias del proyecto

Con `pip-tools` instalado, compilar e instalar:

```bash
pip-compile          # genera requirements.txt desde requirements.in
pip install -r requirements.txt
```

> `pip-compile` solo necesita correr cuando cambian las dependencias, no en cada sesión.

---

## 5. Verificar la instalación

Este proyecto incluye un script de verificación:

```bash
python test_environment.py
```

O de forma manual:

```bash
python -c "import pandas, numpy, sklearn, matplotlib, seaborn, duckdb; print('Entorno OK')"
```

---

## 6. Dependencias actuales del proyecto

Las dependencias directas declaradas en `requirements.in` son:

```
duckdb
jupyter
matplotlib
numpy
openpyxl
pandas
scikit-learn
scipy
seaborn
statsmodels
xlrd
```

---

## 7. Agregar una nueva librería

### Paso 1 — Agregar el paquete a `requirements.in`

Editar el archivo `requirements.in` y agregar el nombre del paquete:

```
duckdb
jupyter
matplotlib
numpy
openpyxl
pandas
scikit-learn
scipy
seaborn
statsmodels
xlrd
nueva-libreria       # <-- agregar acá
```

O desde la terminal:

```bash
echo "nueva-libreria" >> requirements.in
```

### Paso 2 — Recompilar `requirements.txt`

```bash
pip-compile
```

`pip-compile` resuelve las versiones compatibles de **todas** las dependencias (directas y transitivas) y actualiza `requirements.txt`.

### Paso 3 — Instalar

```bash
pip install -r requirements.txt
```

---

## 8. Actualizar una librería existente

### Actualizar un paquete específico

```bash
pip-compile --upgrade-package nombre-paquete
pip install -r requirements.txt
```

Ejemplo — actualizar solo pandas:

```bash
pip-compile --upgrade-package pandas
pip install -r requirements.txt
```

### Actualizar todos los paquetes

```bash
pip-compile --upgrade
pip install -r requirements.txt
```

> `--upgrade` intenta usar la versión más reciente compatible de cada paquete. Verificar que los notebooks no se rompan luego de una actualización masiva.

---

## 9. Eliminar una librería

1. Eliminar la línea correspondiente de `requirements.in`.
2. Recompilar:

```bash
pip-compile
pip install -r requirements.txt
```
