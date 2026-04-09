"""
api.py — API REST para predecir precios de viviendas.

Endpoints:
  GET  /                    → bienvenida
  GET  /health              → estado de la API y el modelo
  POST /predict             → predicción de precio
  GET  /monitor             → resumen de predicciones en producción
  GET  /monitor/pendientes  → predicciones sin precio real asignado
  POST /feedback            → ingresar precios reales de predicciones pasadas
  POST /retrain             → reentrenar con datos de producción confirmados
"""

import joblib
import subprocess
import sys
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List

from src.monitor import (
    inicializar_logs,
    registrar_prediccion,
    guardar_feedback,
    obtener_predicciones_pendientes,
    obtener_resumen,
    TRAINING_DATA,
)


# ── Cargar configuración y modelo ─────────────────────────────────────────────

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

try:
    modelo = joblib.load(cfg["modelo"]["guardar_en"])
    modelo_listo = True
    print(f"Modelo cargado: {cfg['modelo']['guardar_en']}")
except FileNotFoundError:
    modelo = None
    modelo_listo = False
    print("ADVERTENCIA: modelo no encontrado. Ejecuta pipeline.py primero.")

inicializar_logs()

app = FastAPI(
    title="Housing Price API",
    description="Predice precios de viviendas — Boston Housing Dataset",
    version="1.0.0",
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class Vivienda(BaseModel):
    CRIM:    float = Field(..., example=0.00632)
    ZN:      float = Field(..., example=18.0)
    INDUS:   float = Field(..., example=2.31)
    CHAS:    float = Field(..., example=0.0)
    NOX:     float = Field(..., example=0.538)
    RM:      float = Field(..., example=6.575)
    AGE:     float = Field(..., example=65.2)
    DIS:     float = Field(..., example=4.09)
    RAD:     float = Field(..., example=1.0)
    TAX:     float = Field(..., example=296.0)
    PTRATIO: float = Field(..., example=15.3)
    B:       float = Field(..., example=396.9)
    LSTAT:   float = Field(..., example=4.98)


class Prediccion(BaseModel):
    precio_miles_usd: float
    mensaje:          str
    prediction_id:    str


class FeedbackInput(BaseModel):
    """
    Lista de precios reales para las predicciones pendientes.
    Deben estar en el mismo orden que aparecen en GET /monitor/pendientes.
    """
    precios_reales: List[float] = Field(
        ...,
        example=[24.5, 18.3, 31.2],
        description="Precios reales en miles USD, en el mismo orden que las predicciones pendientes"
    )


class Prediccion(BaseModel):
    precio_miles_usd: float
    mensaje:          str
    prediction_id:    str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def raiz():
    return {
        "mensaje": "Housing Price API activa",
        "docs":    "http://127.0.0.1:8000/docs",
    }


@app.get("/health")
def health():
    return {
        "estado":       "ok",
        "modelo_listo": modelo_listo,
        "modelo":       cfg["modelo"]["guardar_en"],
    }


@app.post("/predict", response_model=Prediccion)
def predict(vivienda: Vivienda):
    """Predice el precio de una vivienda en miles de USD."""
    if not modelo_listo:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. Ejecuta pipeline.py primero.",
        )

    features = vivienda.model_dump()
    datos    = pd.DataFrame([features])
    precio   = round(float(modelo.predict(datos)[0]), 2)

    # Registrar en el log de monitoreo
    prediction_id = registrar_prediccion(features, precio)

    return Prediccion(
        precio_miles_usd=precio,
        mensaje=f"Precio estimado: ${precio:.2f}k USD",
        prediction_id=prediction_id,
    )


@app.get("/monitor")
def monitor():
    """Resumen general de predicciones en producción."""
    return obtener_resumen()


@app.get("/monitor/pendientes")
def monitor_pendientes():
    """
    Lista las predicciones que aún no tienen precio real asignado.
    Usa esta lista para saber qué precios reales debes ingresar en /feedback.
    """
    return obtener_predicciones_pendientes()


@app.post("/feedback")
def feedback(data: FeedbackInput):
    """
    Ingresa los precios reales de las predicciones pendientes.

    Los precios deben estar en el mismo orden que aparecen
    en GET /monitor/pendientes.

    Ejemplo:
      Si hay 3 predicciones pendientes, envías:
      {"precios_reales": [24.5, 18.3, 31.2]}
    """
    try:
        resultado = guardar_feedback(data.precios_reales)
        return {
            "estado":  "ok",
            "mensaje": f"{resultado['registros_guardados']} precios reales guardados.",
            "total_datos_para_reentrenar": resultado["total_datos_entrenamiento"],
            "siguiente_paso": "Cuando tengas suficientes datos, usa POST /retrain",
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/retrain")
def retrain(background_tasks: BackgroundTasks):
    """
    Reentrenar el modelo combinando:
      - Dataset original Boston Housing (boston_housing.db)
      - Datos de producción confirmados (training_data.csv)

    Requiere haber ingresado precios reales con POST /feedback primero.
    """
    import os
    n_datos = 0
    if os.path.exists(TRAINING_DATA):
        with open(TRAINING_DATA) as f:
            n_datos = sum(1 for _ in f) - 1

    if n_datos == 0:
        raise HTTPException(
            status_code=400,
            detail="No hay datos de producción confirmados. "
                   "Usa POST /feedback para ingresar precios reales primero.",
        )

    background_tasks.add_task(ejecutar_reentrenamiento)
    return {
        "estado":          "aceptado",
        "datos_nuevos":    n_datos,
        "mensaje":         f"Reentrenamiento iniciado con {n_datos} datos nuevos de producción.",
        "nota":            "Consulta /health para ver cuando el modelo nuevo esté listo.",
    }


def ejecutar_reentrenamiento():
    """Corre el pipeline de entrenamiento y recarga el modelo."""
    global modelo, modelo_listo

    print("\nIniciando reentrenamiento con datos de producción...")
    try:
        resultado = subprocess.run(
            [sys.executable, "pipeline.py", "--version", "retrain_produccion"],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if resultado.returncode == 0:
            modelo       = joblib.load(cfg["modelo"]["guardar_en"])
            modelo_listo = True
            print("Reentrenamiento completado. Modelo recargado.")
        else:
            print(f"Error en reentrenamiento:\n{resultado.stderr}")

    except Exception as e:
        print(f"Error inesperado: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
