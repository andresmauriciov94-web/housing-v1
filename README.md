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
├── Dockerfile        → empaquetado del servicio API
├── docker-compose.yml → orquesta API + MLflow con volúmenes persistentes
├── requirements.txt  → dependencias con versiones mínimas
└── .github/
    └── workflows/
        └── ci.yml    → pipeline CI/CD con GitHub Actions
```

---

## Requisitos

- Python 3.11 o superior
- pip
- Docker (opcional)

---

## Opción A — Ejecución local (sin Docker)

### 1. Clonar el repositorio

```bash
git clone https://github.com/andresmauriciov94-web/housing-v1.git
cd housing-v1
```

### 2. Crear entorno virtual

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Entrenar el modelo

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

### 5. Ver experimentos en MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Abre: http://localhost:5000

### 6. Levantar la API

```bash
python api.py
```

Abre: http://127.0.0.1:8000/docs

---


## Opción B — Solo la API con Docker

Si solo quieres correr la API con docker:

```bash
docker build -t housing-v1 .
docker run -p 8000:8000 housing-v1
```
abre navegador
```bash
http://localhost:8000/docs
```

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
curl -X POST http://localhost:8000/predict \
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

### Feedback loop — reentrenamiento con datos reales

**Paso 1 — Ver predicciones pendientes:**
```bash
curl http://localhost:8000/monitor/pendientes
```

**Paso 2 — Confirmar precios reales en el mismo orden:**
```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"precios_reales": [28.0, 22.5, 31.0]}'
```

**Paso 3 — Reentrenar:**
```bash
curl -X POST http://localhost:8000/retrain
```

El pipeline combina el dataset original con los datos confirmados,
entrena el modelo nuevo y aplica el quality gate.

---

## Quality gate

Cada reentrenamiento compara el RMSE del modelo nuevo contra el actual:

```
RMSE nuevo < RMSE actual  →  modelo reemplazado  (APROBADO)
RMSE nuevo >= RMSE actual →  modelo sin cambios   (RECHAZADO)
```

El resultado queda registrado en MLflow como tag `resultado_quality_gate`.

---

## CI/CD

El repositorio tiene un pipeline de GitHub Actions en cada push a `main`:

```
✓ Verificar sintaxis de todos los módulos
✓ Correr el pipeline de entrenamiento completo
✓ Verificar que la API importa correctamente
✓ Construir imagen Docker
✓ Probar que el contenedor arranca y responde
```

---

## Decisiones técnicas

**Dataset** — OpenML (data_id=531). Persiste en SQLite local.
Los 16 registros con MEDV >= 50 se eliminan por ser valores censurados
artificialmente — justificado visualmente con gráfica en MLflow.

**Escalado selectivo** — StandardScaler en todas las features numéricas
excepto CHAS (variable binaria). El scaler vive dentro del Pipeline de sklearn
eliminando riesgo de data leakage.

**Modelos candidatos** — LinearRegression y Ridge. El selector automático
elige el de menor RMSE. Para agregar un modelo nuevo solo hay que añadirlo
al diccionario `candidatos` en `src/train.py`.

**Quality gate** — El modelo nuevo solo reemplaza al de producción si
tiene menor RMSE. Evita regresiones silenciosas.

**MLflow** — Tracking con SQLite backend. Cada ejecución registra versión,
responsable, métricas, gráficas y resultado del quality gate.

**Feedback loop** — La API registra cada predicción con ID único.
El usuario confirma precios reales con `POST /feedback`.
Esos datos se combinan con el dataset original en el reentrenamiento.

**Docker Compose** — API y MLflow en contenedores separados con red
interna compartida y volúmenes persistentes. MLflow sobrevive reinicios.

---

## Posibles mejoras

- GridSearchCV para Ridge con múltiples valores de alpha
- Agregar RandomForest o XGBoost como candidatos
- Alertas automáticas cuando el RMSE de producción supera un umbral
- Reentrenamiento automático programado
- PostgreSQL como backend de MLflow para producción
- Autenticación en `/retrain` y `/feedback`
- Grafana conectado a un endpoint `/metrics` Prometheus

---

## Bonus Track — Trazabilidad y reproducibilidad

Ver [`docs/traceability.md`](docs/traceability.md) para el análisis de
riesgos de auditoría y propuesta de solución de gobernanza de modelos.

---

## Uso de herramientas AI

Durante el desarrollo se utilizó Claude (Anthropic) como asistente para
generación de boilerplate, revisión de arquitectura y documentación.
Todo el código fue revisado, entendido y validado manualmente.
Las decisiones de diseño son propias del autor.

---

## Dataset

Boston Housing — OpenML versión 1 (data_id=531)
506 observaciones · 13 features · target: precio medio en miles de USD
Se eliminan 16 registros con MEDV >= 50 (valor censurado artificialmente).