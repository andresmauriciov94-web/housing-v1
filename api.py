"""
api.py — API REST para predecir precios de viviendas.

Endpoints:
  GET  /health   → estado de la API
  POST /predict  → predicción de precio

Uso:
  python api.py
"""

import joblib
import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# Cargar configuración y modelo
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


app = FastAPI(
    title="Housing Price API",
    description="Predice precios de viviendas — Boston Housing Dataset",
    version="1.0.0",
)


# ── Esquemas de entrada y salida ──────────────────────────────────────────────

class Vivienda(BaseModel):
    """Características de la vivienda a predecir."""
    CRIM:    float = Field(..., example=0.00632, description="Tasa de criminalidad")
    ZN:      float = Field(..., example=18.0,    description="% suelo residencial")
    INDUS:   float = Field(..., example=2.31,    description="% acres industriales")
    CHAS:    float = Field(..., example=0.0,     description="Límita con río (1=sí)")
    NOX:     float = Field(..., example=0.538,   description="Concentración NO2")
    RM:      float = Field(..., example=6.575,   description="Habitaciones promedio")
    AGE:     float = Field(..., example=65.2,    description="% unidades antiguas")
    DIS:     float = Field(..., example=4.09,    description="Distancia a empleo")
    RAD:     float = Field(..., example=1.0,     description="Acceso a autopistas")
    TAX:     float = Field(..., example=296.0,   description="Tasa de impuesto")
    PTRATIO: float = Field(..., example=15.3,    description="Ratio alumnos/profesor")
    B:       float = Field(..., example=396.9,   description="Índice de población")
    LSTAT:   float = Field(..., example=4.98,    description="% bajo estatus")


class Prediccion(BaseModel):
    """Respuesta con el precio predicho."""
    precio_miles_usd: float
    mensaje:          str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def raiz():
    return {"mensaje": "Housing Price API activa", "docs": "/docs"}


@app.get("/health")
def health():
    return {
        "estado":       "ok",
        "modelo_listo": modelo_listo,
    }


@app.post("/predict", response_model=Prediccion)
def predict(vivienda: Vivienda):
    """Recibe las características de una vivienda y retorna el precio estimado."""

    if not modelo_listo:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. Ejecuta pipeline.py primero.",
        )

    # Convertir input a DataFrame
    datos = pd.DataFrame([vivienda.model_dump()])

    # Predecir
    precio = round(float(modelo.predict(datos)[0]), 2)

    return Prediccion(
        precio_miles_usd=precio,
        mensaje=f"Precio estimado: ${precio:.2f}k USD",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False)
