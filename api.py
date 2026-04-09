"""
api.py — API REST para predecir precios de viviendas.

Endpoints:
  GET  /          → bienvenida
  GET  /health    → estado de la API y el modelo
  POST /predict   → predicción de precio
  GET  /monitor   → resumen de predicciones en producción
  POST /retrain   → dispara reentrenamiento del modelo
"""

import joblib
import subprocess
import sys
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from src.monitor import inicializar_log, registrar_prediccion, obtener_resumen


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

# Inicializar log de producción
inicializar_log()


app = FastAPI(
    title="Housing Price API",
    description="Predice precios de viviendas — Boston Housing Dataset",
    version="1.0.0",
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class Vivienda(BaseModel):
    CRIM:    float = Field(..., example=0.00632, description="Tasa de criminalidad")
    ZN:      float = Field(..., example=18.0,    description="% suelo residencial")
    INDUS:   float = Field(..., example=2.31,    description="% acres industriales")
    CHAS:    float = Field(..., example=0.0,     description="Limita con rio (1=si)")
    NOX:     float = Field(..., example=0.538,   description="Concentracion NO2")
    RM:      float = Field(..., example=6.575,   description="Habitaciones promedio")
    AGE:     float = Field(..., example=65.2,    description="% unidades antiguas")
    DIS:     float = Field(..., example=4.09,    description="Distancia a empleo")
    RAD:     float = Field(..., example=1.0,     description="Acceso a autopistas")
    TAX:     float = Field(..., example=296.0,   description="Tasa de impuesto")
    PTRATIO: float = Field(..., example=15.3,    description="Ratio alumnos/profesor")
    B:       float = Field(..., example=396.9,   description="Indice de poblacion")
    LSTAT:   float = Field(..., example=4.98,    description="% bajo estatus")


class Prediccion(BaseModel):
    precio_miles_usd: float
    mensaje:          str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def raiz():
    return {
        "mensaje": "Housing Price API activa",
        "docs":    "http://127.0.0.1:8000/docs",
    }


@app.get("/health")
def health():
    """Estado de la API y del modelo."""
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
    registrar_prediccion(features, precio)

    return Prediccion(
        precio_miles_usd=precio,
        mensaje=f"Precio estimado: ${precio:.2f}k USD",
    )


@app.get("/monitor")
def monitor():
    """
    Resumen de las predicciones realizadas en producción.
    Muestra estadísticas básicas y alertas detectadas.
    """
    return obtener_resumen()


@app.post("/retrain")
def retrain(background_tasks: BackgroundTasks):
    """
    Dispara el reentrenamiento del modelo en segundo plano.
    La API sigue respondiendo mientras reentrena.
    """
    background_tasks.add_task(ejecutar_reentrenamiento)
    return {
        "estado":  "aceptado",
        "mensaje": "Reentrenamiento iniciado en segundo plano.",
        "nota":    "Consulta /health para ver cuando el modelo nuevo este listo.",
    }


def ejecutar_reentrenamiento():
    """Corre el pipeline de entrenamiento y recarga el modelo."""
    global modelo, modelo_listo

    print("\nIniciando reentrenamiento...")
    try:
        resultado = subprocess.run(
            [sys.executable, "pipeline.py", "--version", "retrain"],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if resultado.returncode == 0:
            # Recargar el modelo nuevo
            modelo       = joblib.load(cfg["modelo"]["guardar_en"])
            modelo_listo = True
            print("Reentrenamiento completado. Modelo recargado.")
        else:
            print(f"Error en reentrenamiento:\n{resultado.stderr}")

    except Exception as e:
        print(f"Error inesperado en reentrenamiento: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False)
