# Bonus Track: Reproducibilidad y Trazabilidad de Modelos

## Contexto

La startup ha recibido una auditoría externa que exige demostrar:
- Qué modelos han estado en producción y en qué períodos
- La capacidad de replicar cualquier predicción pasada, demostrando exactamente
  qué modelo, código y datos estuvieron involucrados en su generación

---

## 1. Análisis del problema y riesgos

### Qué pasa si no podemos demostrar trazabilidad

**Riesgo regulatorio**

En mercados inmobiliarios regulados, las decisiones automatizadas que afectan
a usuarios deben ser explicables y auditables. Si la empresa no puede demostrar
qué modelo generó una predicción específica, ni con qué datos fue entrenado:

- Multas por incumplimiento regulatorio
- Suspensión de operaciones hasta regularizar
- Obligación de rever decisiones pasadas sin poder identificarlas

**Riesgo operacional**

Sin trazabilidad, ante un modelo defectuoso en producción:

- No se puede identificar desde cuándo está fallando
- No se puede hacer rollback a una versión anterior conocida
- No se puede reproducir el error para diagnosticarlo
- La recuperación es manual, lenta y propensa a nuevos errores

**Riesgo reputacional**

Una auditoría fallida es pública. Para una startup que busca escalar:

- Clientes corporativos exigen evidencia de gobernanza antes de contratar
- Inversores interpretan la falta de trazabilidad como deuda técnica grave
- La cobertura mediática negativa es difícil de revertir

**Riesgo de deuda técnica**

Con 50 versiones de modelos sin registro:

- El equipo no puede responder preguntas básicas: con qué datos entrenamos
  el modelo de marzo, por qué lo reemplazamos
- La experimentación se vuelve caótica
- Onboarding de nuevos ingenieros es casi imposible sin historial

---

## 2. Propuesta de solución general

### Principio central

Cualquier predicción debe ser reproducible conociendo tres elementos:
el código exacto + los datos exactos + los parámetros exactos.

La solución se implementa en cuatro capas:

---

### Capa 1 — Git como fuente de verdad del código

Cada cambio al pipeline queda en un commit con hash único e inmutable.
Ese hash se registra automáticamente en MLflow junto con cada experimento.

Dado un modelo en producción, siempre se puede hacer checkout del código
exacto que lo generó con git checkout hash_del_commit.

---

### Capa 2 — MLflow como registro de experimentos y modelos

Cada ejecución del pipeline registra automáticamente:

- Versión del experimento y responsable
- Hash del commit de Git
- Parámetros completos del modelo
- Métricas de entrenamiento y evaluación
- Artefactos: modelo serializado, gráficas
- Resultado del quality gate
- Timestamp de inicio y fin

El Model Registry mantiene el historial completo de qué versión estuvo
activa y en qué fechas. Para responder a la auditoría basta con consultar
ese registro.

---

### Capa 3 — Logs estructurados de inferencia

Cada predicción registra prediction_id, timestamp, model_version,
input_features y precio_predicho.

Con esos datos se puede reconstruir cualquier predicción pasada:

1. Identificar prediction_id en el log
2. Obtener model_version en MLflow Registry
3. Obtener run_id y git_commit del experimento
4. Checkout del commit y mismo dataset
5. Predicción reproducida de forma determinista

---

### Capa 4 — Quality gate como mecanismo de gobernanza

El modelo nuevo solo llega a producción si supera el umbral de calidad.
Cada transición de modelo en el Registry fue una decisión controlada
y medida, registrada con métricas objetivas.

---

### Diagrama de trazabilidad completa

```
Desarrollador hace cambio
        |
git commit (hash: f8e3a1b)
        |
python pipeline.py --version v2.0
        |
MLflow registra run_id, git_commit, params, metrics, artifacts
        |
Quality gate: APROBADO
        |
Model Registry: version 2 activa desde 2026-04-09
        |
API carga version 2
        |
Prediccion: ID=20260409101502, precio=29.35
        |
Log: prediction_id + model_version + inputs + output

--- AUDITORIA ---
"Reproduzca la prediccion ID=20260409101502"
        |
Log → model_version=2 → run_id=abc123
        |
MLflow → git_commit=f8e3a1b
        |
git checkout f8e3a1b + mismo dataset + mismos params
        |
Prediccion reproducida: 29.35 OK
```

---

### Por qué esta solución es adecuada

**Escalabilidad** — MLflow maneja miles de experimentos sin degradación.
El backend puede migrar de SQLite a PostgreSQL sin cambiar el código.
Los artefactos pueden moverse a MinIO para almacenamiento distribuido.

**Reproducibilidad** — La combinación git_commit + dataset + random_seed
hace que cualquier experimento sea reproducible de forma determinista,
incluso años después. No depende de la memoria del equipo ni de
documentación manual que se vuelve obsoleta.

**Gobernanza** — El quality gate automatizado reemplaza procesos manuales
que no escalan. El Model Registry con historial de versiones responde
directamente las preguntas de una auditoría sin intervención humana.

**Sin vendor lock-in** — Toda la solución es open-source y self-hosted.
MLflow, SQLite, Git — sin dependencia de servicios de nube específicos.
El sistema es portable a cualquier infraestructura.

---

### Lo que NO es suficiente sin esta solución

- Guardar solo el .pkl del modelo: sin el código y los datos la
  reproducibilidad no es posible
- Versionado solo en Git: no registra los datos ni los hiperparámetros
  de cada experimento
- Logs en texto libre: no permiten consultas eficientes para auditorías
- Nombres de archivo con fecha: no hay trazabilidad entre el archivo
  y las predicciones que hizo en producción
