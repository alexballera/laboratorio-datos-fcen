# 🤖 Guía del Agente Tutor-Profesor IA

## ¿Qué es?

El **agente tutor-profesor** es un asistente de inteligencia artificial especializado en el curso de Laboratorio de Datos de la FCEN-UBA. Está diseñado para ayudarte a aprender Python, Pandas, SQL, Machine Learning y análisis de datos de forma didáctica.

## ¿Qué puede hacer por ti?

### ✅ Explicar conceptos

**Ejemplo:**
```
/tutor-profesor ¿Qué es un GroupBy en Pandas?
/tutor-profesor explícame regresión lineal simple
/tutor-profesor diferencia entre INNER JOIN y LEFT JOIN
```

El tutor te dará:
- Definición clara y simple
- Analogía o ejemplo cotidiano
- Código de ejemplo
- Errores comunes a evitar

### ✅ Revisar tu código

**Ejemplo:**
```
/tutor-profesor revisa mi notebook de la Clase 5

/tutor-profesor ¿está bien este código?
[pegar tu código aquí]
```

El tutor evaluará:
- Correctitud del código
- Buenas prácticas
- Claridad y legibilidad
- Sugerencias de mejora

### ✅ Ayudarte con errores

**Ejemplo:**
```
/tutor-profesor me da este error: KeyError: 'columna'

/tutor-profesor ¿por qué mi groupby no funciona?
```

El tutor:
- Explicará qué significa el error
- Te mostrará cómo solucionarlo
- Te dará un ejemplo correcto
- Te sugerirá cómo evitarlo en el futuro

### ✅ Proponer ejercicios

**Ejemplo:**
```
/tutor-profesor dame ejercicios de Pandas nivel básico
/tutor-profesor quiero practicar joins en SQL
/tutor-profesor ejercicios de K-NN
```

El tutor te dará:
- Ejercicios graduados por dificultad
- Consignas claras
- Pistas opcionales (puedes decidir si las lees)
- Solución comentada

### ✅ Guiarte en trabajos prácticos

**Ejemplo:**
```
/tutor-profesor necesito ayuda con el TP1
/tutor-profesor ¿qué enfoque uso para este problema? [descripción]
```

El tutor:
- NO te dará la solución directa
- Te hará preguntas guía para que razones
- Te sugerirá qué conceptos revisar
- Te orientará hacia el enfoque correcto

### ✅ Evaluar tus trabajos

**Ejemplo:**
```
/tutor-profesor evalúa mi TP con la rúbrica del curso
```

El tutor usará criterios claros:
- Correctitud (40%)
- Metodología (30%)
- Claridad (20%)
- Prolijidad (10%)

## Cómo usarlo

### Opción 1: GitHub Copilot en VS Code (recomendado)

1. Instala la extensión GitHub Copilot en VS Code
2. Abre este repositorio en VS Code
3. Abre el chat de Copilot (Ctrl+I o Cmd+I)
4. Escribe `/tutor-profesor` seguido de tu consulta

**Ejemplo:**
```
/tutor-profesor ¿cómo filtro filas en Pandas donde la edad sea mayor a 18?
```

### Opción 2: Chat de GitHub Copilot

1. Abre el panel lateral de Copilot Chat
2. Escribe `/tutor-profesor` y tu pregunta
3. El agente responderá con explicaciones didácticas

## Ejemplos de consultas útiles

### Para principiantes

```
/tutor-profesor ¿cómo empiezo con Pandas?
/tutor-profesor instalación del entorno Python del curso
/tutor-profesor ¿qué es un DataFrame?
/tutor-profesor primeros pasos en Jupyter Notebook
```

### Para práctica

```
/tutor-profesor ejercicios de la Clase 1
/tutor-profesor dame un dataset para practicar groupby
/tutor-profesor quiero practicar visualización con matplotlib
```

### Para resolver dudas

```
/tutor-profesor diferencia entre loc e iloc
/tutor-profesor cuándo usar WHERE y cuándo usar HAVING en SQL
/tutor-profesor ¿qué métrica usar para clasificación desbalanceada?
```

### Para trabajos prácticos

```
/tutor-profesor ¿cómo abordo el análisis exploratorio del TP?
/tutor-profesor ¿está bien mi estrategia de limpieza de datos?
/tutor-profesor ¿qué modelo de ML sería apropiado para este problema?
```

## Características del tutor

### 🇦🇷 Siempre en español

El tutor responde **siempre en español latinoamericano neutro**. Todos los ejemplos, explicaciones y comentarios de código están en español.

### 📚 Pedagógico

El tutor no solo da respuestas, **enseña**:
- Explica el "por qué", no solo el "cómo"
- Propone que pruebes código, no solo leer
- Plantea preguntas para que razones
- Da feedback constructivo

### 🎯 Enfocado en el curso

El tutor conoce:
- Todo el contenido del curso (Clases 0-19)
- Los datasets que usamos
- Los ejercicios y TPs
- Las convenciones del curso

### 🚫 Límites del tutor

El tutor **NO**:
- Da soluciones completas de TPs (te guía, no resuelve por ti)
- Reemplaza la consulta con el profesor
- Cubre temas fuera del programa del curso

## Consejos para aprovechar al máximo

### ✅ Haz preguntas específicas

**Mejor:**
```
/tutor-profesor ¿cómo calculo el promedio de una columna agrupando por otra en Pandas?
```

**Menos efectivo:**
```
/tutor-profesor ayuda con Pandas
```

### ✅ Muestra tu intento primero

Cuando pidas ayuda con un error, muestra qué intentaste:

```
/tutor-profesor intenté esto:
df.groupby('ciudad').sum()
Pero me da error "cannot sum strings". ¿Cómo lo arreglo?
```

### ✅ Pide profundización si es necesario

Si una explicación no te queda clara:

```
/tutor-profesor ¿podés dar un ejemplo más simple de esto?
/tutor-profesor no entendí la diferencia, ¿podés explicar con una analogía?
```

### ✅ Usa el tutor para practicar

No solo cuando tengas dudas:

```
/tutor-profesor dame 3 ejercicios de dificultad creciente sobre merge en Pandas
```

## Preguntas frecuentes

### ¿El tutor puede corregir mis TPs?

El tutor puede **revisar** tu trabajo y darte feedback, pero la calificación oficial la pone el profesor.

### ¿Puedo confiar en las respuestas del tutor?

El tutor está diseñado para seguir las mejores prácticas del curso, pero siempre:
- Prueba el código que te sugiere
- Contrasta con la documentación oficial
- Consulta al profesor si tienes dudas

### ¿El tutor puede revisar mi notebook completo?

Sí, pero es mejor si:
1. Primero ejecutas todo sin errores
2. Le indicas qué aspectos específicos revisar
3. Le muestras secciones particulares si es muy largo

### ¿Puedo usar el tutor en parciales?

**No.** El tutor es para aprender durante las prácticas, no para evaluaciones formales. Consulta las normas del curso con el profesor.

## Recursos adicionales

- [SKILL.md](.github/skills/tutor-profesor/SKILL.md): Documentación completa del tutor
- [README.md](README.md): Guía general del repositorio
- [AGENTS.md](AGENTS.md): Instrucciones para agentes IA

## Soporte

Si el tutor no funciona o tiene comportamientos inesperados:
1. Verifica que estás usando la sintaxis `/tutor-profesor` correcta
2. Intenta reiniciar el chat de Copilot
3. Reporta el problema en [Issues](https://github.com/alexballera/laboratorio-fcen/issues)

---

**¡Aprovecha este recurso para acelerar tu aprendizaje! 🚀**
