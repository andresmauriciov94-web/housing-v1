# Housing Price Prediction — MLOps Pipeline

Predicción de precios de viviendas usando el dataset Boston Housing.
Pipeline completo de MLOps con entrenamiento, experimentación, API REST,
monitoreo, feedback loop y reentrenamiento continuo.
Stack 100% open-source · sin dependencia de ningún proveedor cloud.

---

## Estructura del proyecto

```
housing-v1/
├── src/
│   ├── data.py       → descarga, limpieza y gráfica de distribución MEDV
│   ├── train.py      → entrenamiento, quality gate y registro en MLflow
│   ├── monitor.py    → registro de predicciones y feedback de precios reales
│   └── dashboard.py  → dashboard HTML de monitoreo en producción
├── pipeline.py       → orquestador del ciclo de entrenamiento
├── api.py            → API REST con FastAPI
├── deploy.py         → script de despliegue con validaciones previas
├── config.yaml       → parámetros del proyecto
├── Dockerfile        → empaquetado del servicio
├── requirements.txt  → dependencias con versiones fijadas
└── .github/
    └── workflows/
        └── ci.yml    → pipeline CI/CD con GitHub Actions
```

---

## Requisitos

- Python 3.11
- pip

---

## Instalación

**1. Clonar el repositorio:**

```bash
git clone https://github.com/andresmauriciov94-web/housing-v1.git
cd housing-v1
```

**2. Crear entorno virtual:**

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

**3. Instalar dependencias:**

```bash
pip install -r requirements.txt
```

---

## Uso

### Paso 1 — Entrenar el modelo

```bash
python pipeline.py --version v1.0 --responsable TuNombre
```

El pipeline ejecuta en orden:

```
[1/2] Descarga Boston Housing desde OpenML y persiste en SQLite
      Genera gráfica de distribución MEDV → MLflow
[2/2] Entrena candidatos (LinearRegression + Ridge)
      Aplica quality gate → solo reemplaza si el nuevo modelo es mejor
      Guarda modelo_final.pkl
```

Al finalizar verás:

```
PIPELINE COMPLETADO
Versión         : v1.0
RMSE            : 3.28
R2              : 0.78
MAE             : 2.14
Modelo guardado : SI
```

### Paso 2 — Ver experimentos en MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Abre en el navegador: http://localhost:5000

Verás por cada experimento:
- Métricas: RMSE, R2, MAE de cada candidato
- Tags: versión, responsable, quality gate resultado
- Artifacts: gráfica de distribución MEDV, modelos entrenados

### Paso 3 — Levantar la API

```bash
python api.py
```

O con validación previa del modelo:

```bash
python deploy.py
```

API disponible en: http://127.0.0.1:8000/docs

---

## Endpoints

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/` | Bienvenida |
| GET | `/health` | Estado de la API y el modelo |
| POST | `/predict` | Predicción de precio |
| GET | `/dashboard` | Dashboard visual de monitoreo |
| GET | `/monitor` | Resumen JSON de predicciones |
| GET | `/monitor/pendientes` | Predicciones sin precio real asignado |
| POST | `/feedback` | Ingresar precios reales de predicciones pasadas |
| POST | `/retrain` | Reentrenar con datos de producción confirmados |

### Predicción

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "CRIM": 0.00632, "ZN": 18.0, "INDUS": 2.31, "CHAS": 0.0,
    "NOX": 0.538, "RM": 6.575, "AGE": 65.2, "DIS": 4.09,
    "RAD": 1.0, "TAX": 296.0, "PTRATIO": 15.3, "B": 396.9,
    "LSTAT": 4.98
  }'
```

Respuesta:

```json
{
  "precio_miles_usd": 29.35,
  "mensaje": "Precio estimado: $29.35k USD",
  "prediction_id": "20260409101502523016"
}
```

### Dashboard visual

```
http://127.0.0.1:8000/dashboard
```

Muestra:
- Total de predicciones, precio promedio, mínimo, máximo
- Alertas detectadas y predicciones pendientes de feedback
- Gráfica de precios en el tiempo
- Histograma de distribución de precios
- Tabla de las últimas 10 predicciones

### Feedback loop — reentrenamiento con datos reales

**Paso 1 — Ver predicciones pendientes:**

```bash
curl http://127.0.0.1:8000/monitor/pendientes
```

**Paso 2 — Ingresar precios reales en el mismo orden:**

```bash
curl -X POST http://127.0.0.1:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"precios_reales": [28.0, 22.5, 31.0]}'
```

**Paso 3 — Reentrenar con esos datos:**

```bash
curl -X POST http://127.0.0.1:8000/retrain
```

El pipeline combina el dataset original con los datos de producción confirmados,
entrena el modelo nuevo y aplica el quality gate — solo reemplaza si mejora el RMSE.

---

## Quality gate

Cada reentrenamiento compara el RMSE del modelo nuevo contra el modelo en producción:

```
RMSE nuevo < RMSE actual  →  modelo reemplazado  (APROBADO)
RMSE nuevo >= RMSE actual →  modelo sin cambios   (RECHAZADO)
```

El resultado queda registrado en MLflow como tag `resultado_quality_gate`.

---

## Con Docker

```bash
# Construir la imagen
docker build -t housing-api .

# Correr el contenedor
docker run -p 8000:8000 housing-api

# Verificar
curl http://localhost:8000/health
```

---

## CI/CD

El repositorio tiene un pipeline de GitHub Actions que se activa en cada push a `main`:

```
✓ Verificar sintaxis de todos los módulos
✓ Correr el pipeline de entrenamiento completo
✓ Verificar que la API importa correctamente
✓ Construir imagen Docker
✓ Probar que el contenedor arranca y responde
```

---

## Decisiones técnicas

**Dataset** — Se usa la versión de OpenML (data_id=531) en lugar de la deprecada
de scikit-learn. Se persiste en SQLite local para no descargar en cada ejecución.
Los 16 registros con MEDV >= 50 se eliminan por ser valores censurados artificialmente
en el dataset original — justificado visualmente con gráfica en MLflow.

**Escalado selectivo** — Se aplica StandardScaler a todas las features numéricas
excepto CHAS, que es una variable binaria categórica (0/1) y no se beneficia del escalado.
El scaler queda dentro del Pipeline de sklearn, eliminando cualquier riesgo de data leakage.

**Modelos candidatos** — Se entrenan LinearRegression y Ridge en cada ejecución.
El selector automático elige el de menor RMSE. Para agregar un nuevo modelo basta
con añadirlo al diccionario `candidatos` en `src/train.py`.

**Quality gate** — El modelo nuevo solo reemplaza al de producción si tiene menor RMSE.
Esto evita que un reentrenamiento con pocos datos nuevos degrade el modelo en producción.

**MLflow** — Tracking con SQLite backend (sin servidor adicional). Cada ejecución
registra versión, responsable, métricas, gráficas y resultado del quality gate.
Permite comparar experimentos y auditar decisiones.

**Feedback loop** — La API registra cada predicción con un ID único. El usuario
puede confirmar el precio real de cada predicción mediante `POST /feedback`.
Esos datos se combinan con el dataset original en el siguiente reentrenamiento.

**FastAPI** — Validación automática de inputs con Pydantic, documentación Swagger
generada automáticamente, manejo de errores con códigos HTTP correctos.

**Docker** — Imagen basada en python:3.11-slim para mantenerla liviana.
El modelo se incluye en la imagen para garantizar portabilidad sin necesidad de entrenar.

**CI/CD** — GitHub Actions valida sintaxis, corre el pipeline completo y prueba
el contenedor Docker en cada push. Sin configuración adicional.

---

## Posibles mejoras

- GridSearchCV para Ridge con múltiples valores de alpha
- Agregar RandomForest o XGBoost como candidatos adicionales
- Alertas automáticas cuando el RMSE de producción supera un umbral
- Reentrenamiento automático programado (cron job)
- PostgreSQL como backend de MLflow para entornos de producción
- Autenticación en endpoints sensibles como `/retrain` y `/feedback`
- Grafana conectado a `/metrics` para dashboards operacionales

---

## Bonus Track — Trazabilidad y reproducibilidad

Ver [`docs/traceability.md`](docs/traceability.md) para el análisis completo
de riesgos de auditoría y propuesta de solución de gobernanza de modelos.

---

## Uso de herramientas AI

Durante el desarrollo se utilizó Claude (Anthropic) como asistente para:
generación de boilerplate, revisión de decisiones de arquitectura y documentación.
Todo el código fue revisado, entendido y validado manualmente.
Las decisiones de diseño son propias del autor.

---

## Dataset

Boston Housing — OpenML versión 1 (data_id=531)
506 observaciones · 13 features · target: precio medio en miles de USD
Se eliminan 16 registros con MEDV >= 50 (valor censurado artificialmente).
