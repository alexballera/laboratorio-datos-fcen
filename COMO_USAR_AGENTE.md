# ✅ Agente Tutor-Profesor - Instalación Completa

## 🎯 Resumen de lo Creado

Se ha creado exitosamente el **agente tutor-profesor** para el curso de Laboratorio de Datos (FCEN-UBA) que aparece en el **selector de agentes** de VS Code, tal como se muestra en la imagen que compartiste.

---

## 📁 Archivos Creados

### 1. Agente Personalizado (aparece en el menú)
```
.github/agents/tutor-profesor.agent.md (254 líneas)
```
Este archivo hace que el agente aparezca en el selector desplegable de VS Code.

### 2. Skill Complementario (comando slash)
```
.github/skills/tutor-profesor/SKILL.md (364 líneas)
```
Documentación extendida accesible con `/tutor-profesor`.

### 3. Referencias Adicionales
```
.github/skills/tutor-profesor/references/
├── ejercicios-tipo.md     # Ejercicios modelo
├── errores-comunes.md     # Catálogo de errores
└── datasets.md            # Guía de datasets
```

### 4. Documentación para Estudiantes
```
GUIA_TUTOR.md (261 líneas)     # Guía de uso del tutor
README.md (457 líneas)         # Renovado completamente
RESUMEN_TRABAJO.md             # Este documento de resumen
```

---

## 🚀 Cómo Verificar que Funciona

### Paso 1: Reiniciar VS Code
Para que VS Code detecte el nuevo agente:
1. Cierra VS Code completamente
2. Vuelve a abrir el repositorio

### Paso 2: Abrir el Selector de Agentes
1. Presiona **Ctrl+Shift+I** (o Cmd+Shift+I en Mac) para abrir el chat de Copilot
2. En la parte superior del chat, verás un **menú desplegable** con opciones como:
   - Agent
   - Ask
   - Plan
   - **Tutor Laboratorio Datos** ← **¡Tu nuevo agente aparecerá aquí!**

### Paso 3: Seleccionar el Agente
1. Haz clic en el menú desplegable
2. Selecciona **"Tutor Laboratorio Datos"**
3. Escribe tu primera consulta, por ejemplo:

```
Hola, ¿qué temas del curso puedes ayudarme a aprender?
```

### Paso 4: Verificar que Responde en Español
El agente debe responder:
- ✅ En español latinoamericano
- ✅ Con tono didáctico y accesible
- ✅ Mostrando conocimiento del curso (5 módulos)

---

## 📊 Estructura Final del Proyecto

```
laboratorio-fcen/
│
├── README.md                          # ✨ Renovado (457 líneas)
├── GUIA_TUTOR.md                      # ✨ Nuevo (261 líneas)
├── RESUMEN_TRABAJO.md                 # ✨ Nuevo
├── COMO_USAR_AGENTE.md                # ✨ Nuevo (este archivo)
│
└── .github/
    ├── agents/
    │   └── tutor-profesor.agent.md   # ✨ Agente personalizado (254 líneas)
    │
    └── skills/
        └── tutor-profesor/            # ✨ Skill complementario
            ├── SKILL.md               # 364 líneas
            └── references/
                ├── ejercicios-tipo.md
                ├── errores-comunes.md
                └── datasets.md
```

---

## 🎮 Dos Formas de Usar el Tutor

### Opción 1: Selector de Agentes (Recomendado)

**Cuándo usar**: Para conversaciones largas sobre el curso

**Cómo**:
1. Ctrl+Shift+I para abrir chat
2. Clic en el menú desplegable
3. Seleccionar "Tutor Laboratorio Datos"
4. Escribir preguntas normalmente (sin prefijos)

**Ejemplo de conversación**:
```
Usuario: ¿Cómo hago un groupby en Pandas?
Tutor: [Respuesta didáctica]

Usuario: Dame un ejemplo con el dataset de arbolado
Tutor: [Ejemplo específico del curso]

Usuario: ¿Y si quiero agrupar por dos columnas?
Tutor: [Extensión del concepto]
```

**Ventaja**: El agente permanece activo, entiende el contexto de la conversación.

---

### Opción 2: Comando Slash

**Cuándo usar**: Para consultas puntuales rápidas

**Cómo**:
```
/tutor-profesor ¿qué es un DataFrame?
```

**Ventaja**: No necesitas cambiar de agente si estás usando otro.

---

## 🔍 Qué Puede Hacer el Tutor

### ✅ Explicar Conceptos
```
Explícame normalización de bases de datos
¿Qué diferencia hay entre INNER JOIN y LEFT JOIN?
¿Cómo funciona K-Means clustering?
```

### ✅ Revisar Código
```
Revisa este código de groupby:
[pegar código]

¿Está bien esta limpieza de datos?
[mostrar código]
```

### ✅ Proponer Ejercicios
```
Dame ejercicios de Pandas nivel básico
Necesito practicar SQL joins
Ejercicios de clasificación con árboles de decisión
```

### ✅ Corregir Errores
```
Me da este error: KeyError: 'ciudad'
¿Por qué mi modelo tiene accuracy muy baja?
```

### ✅ Guiar en TPs (sin dar la solución)
```
Estoy en el TP de Titanic, ¿cómo empiezo?
¿Qué features debería usar para este modelo?
```

---

## 🎓 Características del Agente

- **Idioma**: Siempre español (código, comentarios, explicaciones)
- **Tono**: Didáctico, claro, motivador
- **Cobertura**: Los 5 módulos del curso (Python → ML No Supervisado)
- **Metodología**: Scaffolding, aprendizaje activo, feedback constructivo
- **Datasets**: Conoce todos los datasets del curso (arbolado, Titanic, Iris, tips, etc.)

---

## 🧪 Pruebas Sugeridas

Prueba estas consultas para verificar que el agente funciona correctamente:

### Prueba 1: Conocimiento del Curso
```
¿Qué temas se cubren en la Clase 5?
```
**Esperado**: Debe mencionar SQL, álgebra relacional, DuckDB.

### Prueba 2: Explicación de Conceptos
```
Explícame qué es un groupby en Pandas con un ejemplo simple
```
**Esperado**: Definición, analogía, código de ejemplo, cuándo usarlo.

### Prueba 3: Revisión de Código
```
Revisa este código:
df.groupby('ciudad').sum()

Me da error de que no puede sumar strings
```
**Esperado**: Explicación del error, solución correcta, ejemplo.

### Prueba 4: Ejercicios
```
Dame un ejercicio básico de filtrado en Pandas
```
**Esperado**: Ejercicio con consigna clara, datos, solución opcional.

### Prueba 5: Datasets del Curso
```
¿Qué dataset usamos para practicar clasificación?
```
**Esperado**: Debe mencionar Titanic e Iris, con su estructura.

---

## 🐛 Solución de Problemas

### El agente no aparece en el selector

**Soluciones**:
1. Reinicia VS Code completamente
2. Verifica que GitHub Copilot esté activo (ícono en la barra inferior)
3. Verifica que el archivo existe:
   ```bash
   ls -l .github/agents/tutor-profesor.agent.md
   ```
4. Verifica que el frontmatter YAML está bien formado:
   ```bash
   head -10 .github/agents/tutor-profesor.agent.md
   ```

### El agente responde en inglés

Si el agente responde en inglés, recuérdale:
```
Por favor responde en español, es parte de tus instrucciones del curso
```

### El agente no conoce el curso

Verifica que el archivo `.agent.md` tenga la sección de "Conocimiento del Curso" con los 5 módulos.

---

## 📚 Próximos Pasos

1. **Probar el agente** siguiendo las pruebas sugeridas arriba
2. **Compartir con estudiantes** el [README.md](README.md) y [GUIA_TUTOR.md](GUIA_TUTOR.md)
3. **Recopilar feedback** de estudiantes sobre qué funciona bien y qué mejorar
4. **Iterar el agente** basándose en casos de uso reales
5. **Expandir referencias** con más ejercicios y ejemplos

---

## 📝 Documentación Relacionada

- [README.md](README.md): Guía completa del repositorio
- [GUIA_TUTOR.md](GUIA_TUTOR.md): Guía específica del tutor para estudiantes
- [AGENTS.md](AGENTS.md): Instrucciones generales para agentes IA
- [RESUMEN_TRABAJO.md](RESUMEN_TRABAJO.md): Resumen técnico del trabajo realizado

---

## ✅ Checklist de Verificación

Marca cada ítem cuando lo verifiques:

- [ ] El agente aparece en el selector de VS Code
- [ ] Responde en español a todas las consultas
- [ ] Conoce los 5 módulos del curso
- [ ] Puede revisar código y dar feedback
- [ ] Propone ejercicios cuando se le pide
- [ ] Guía en TPs sin dar soluciones completas
- [ ] Los estudiantes pueden usar [GUIA_TUTOR.md](GUIA_TUTOR.md) para aprender a usarlo

---

**¡El agente tutor-profesor está listo para ayudar a los estudiantes! 🚀📊**

**Fecha**: 27 de marzo de 2026  
**Autor**: Agente IA colaborando con Alexander Ballera
