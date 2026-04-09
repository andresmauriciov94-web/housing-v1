# Housing Price Prediction — MLOps Pipeline

Predicción de precios de viviendas usando el dataset Boston Housing.  
Pipeline completo de MLOps con entrenamiento, registro de experimentos, API REST y monitoreo.  
Stack 100% open-source · sin dependencia de ningún proveedor cloud.

---

## Estructura del proyecto

```
housing-v1/
├── src/
│   ├── data.py       → descarga y limpieza del dataset
│   ├── train.py      → entrenamiento y registro en MLflow
│   └── monitor.py    → monitoreo de predicciones en producción
├── pipeline.py       → orquestador del ciclo de entrenamiento
├── api.py            → API REST con FastAPI
├── deploy.py         → script de despliegue con validaciones
├── config.yaml       → parámetros del proyecto
├── Dockerfile        → empaquetado del servicio
├── requirements.txt  → dependencias
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
[1/3] Descarga Boston Housing desde OpenML y persiste en SQLite
[2/3] Limpia datos, entrena modelo lineal y registra en MLflow
[3/3] Guarda modelo_final.pkl listo para la API
```

Al finalizar verás:

```
COMPLETADO
Versión : v1.0
RMSE    : 3.28
R2      : 0.78
MAE     : 2.14
```

### Paso 2 — Ver experimentos en MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Abre en el navegador: http://localhost:5000

Verás cada experimento con sus métricas, parámetros y el modelo registrado.

### Paso 3 — Levantar la API

```bash
python api.py
```

O usando el script de despliegue (verifica el modelo antes de arrancar):

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
| GET | `/monitor` | Resumen de predicciones en producción |
| POST | `/retrain` | Dispara reentrenamiento del modelo |

### Ejemplo de predicción

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
  "precio_miles_usd": 24.5,
  "mensaje": "Precio estimado: $24.50k USD"
}
```

### Ejemplo de monitoreo

```bash
curl http://127.0.0.1:8000/monitor
```

Respuesta:

```json
{
  "total_predicciones": 10,
  "precio_promedio": 22.3,
  "precio_minimo": 14.1,
  "precio_maximo": 38.7,
  "total_alertas": 0,
  "estado": "OK"
}
```

### Reentrenamiento

```bash
curl -X POST http://127.0.0.1:8000/retrain
```

Dispara el pipeline de entrenamiento en segundo plano. La API sigue respondiendo mientras reentrena. Al finalizar carga el nuevo modelo automáticamente.

---

## Con Docker

```bash
# Construir la imagen
docker build -t housing-api .

# Correr el contenedor
docker run -p 8000:8000 housing-api

# Probar
curl http://localhost:8000/health
```

---

## CI/CD

El repositorio tiene un pipeline de GitHub Actions que se activa en cada push a `main`:

```
✓ Verificar sintaxis del código
✓ Correr el pipeline de entrenamiento completo
✓ Verificar que la API importa correctamente
✓ Construir imagen Docker
✓ Probar que el contenedor arranca y responde
```

---

## Decisiones técnicas

**Dataset** — Se usa la versión de OpenML (data_id=531) en lugar de la deprecada de scikit-learn. Se persiste en SQLite local para no descargar en cada ejecución.

**Modelo base** — Regresión lineal con StandardScaler como punto de partida. Simple, interpretable y reproducible. La arquitectura permite agregar modelos más complejos sin cambiar el pipeline.

**MLflow** — Tracking con SQLite backend (sin servidor adicional). Cada ejecución registra versión, responsable, métricas y el modelo. Permite comparar experimentos y hacer rollback.

**FastAPI** — Validación automática de inputs con Pydantic, documentación Swagger generada automáticamente, manejo de errores con códigos HTTP correctos.

**Monitoreo** — Log CSV de cada predicción con detección de anomalías por rango de precio. Simple y sin dependencias adicionales.

**Docker** — Imagen basada en python:3.11-slim para mantenerla liviana. El modelo se incluye en la imagen para garantizar portabilidad.

**CI/CD** — GitHub Actions valida sintaxis, corre el pipeline completo y prueba el contenedor Docker en cada push. Sin configuración adicional.

---

## Posibles mejoras

- Agregar modelos adicionales (Ridge, RandomForest, XGBoost) y selección automática del mejor
- Reentrenamiento automático cuando el RMSE de producción supera un umbral
- Base de datos PostgreSQL como backend de MLflow para producción
- Autenticación en endpoints sensibles como `/retrain`
- Dashboard de monitoreo con Grafana

---

## Uso de herramientas AI

Durante el desarrollo se utilizó Claude (Anthropic) como asistente para generación de boilerplate, revisión de decisiones de arquitectura y documentación. Todo el código fue revisado, entendido y validado manualmente. Las decisiones de diseño son propias del autor.

---

## Dataset

Boston Housing — OpenML versión 1 (data_id=531)  
506 observaciones · 13 features · target: precio medio en miles de USD  
Se eliminan registros con MEDV >= 50 (valor censurado artificialmente).
