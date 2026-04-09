"""
src/monitor.py — Monitoreo básico del modelo en producción.

Registra cada predicción en un archivo CSV para:
  - Detectar anomalías en los inputs
  - Ver la distribución de predicciones en el tiempo
  - Disparar alertas si el modelo se comporta raro
"""

import csv
import os
from datetime import datetime


LOG_PATH = "produccion_log.csv"

# Columnas del log
COLUMNAS = [
    "timestamp",
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM",
    "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT",
    "precio_predicho",
    "alerta",
]

# Rango normal del precio predicho (en miles USD)
PRECIO_MIN = 5.0
PRECIO_MAX = 50.0


def inicializar_log():
    """Crea el archivo de log si no existe."""
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(COLUMNAS)
        print(f"Log de produccion creado: {LOG_PATH}")


def registrar_prediccion(features: dict, precio: float):
    """
    Registra una predicción en el log CSV.
    Si el precio está fuera del rango normal, marca una alerta.
    """
    alerta = ""

    # Detectar precio fuera de rango
    if precio < PRECIO_MIN:
        alerta = f"PRECIO_BAJO ({precio:.2f} < {PRECIO_MIN})"
    elif precio > PRECIO_MAX:
        alerta = f"PRECIO_ALTO ({precio:.2f} > {PRECIO_MAX})"

    # Registrar en CSV
    fila = {
        "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "precio_predicho": precio,
        "alerta":          alerta,
        **features,
    }

    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNAS)
        writer.writerow(fila)

    if alerta:
        print(f"  ALERTA: {alerta}")


def obtener_resumen() -> dict:
    """
    Lee el log y retorna un resumen estadístico básico.
    Útil para el endpoint /monitor.
    """
    if not os.path.exists(LOG_PATH):
        return {"total_predicciones": 0, "mensaje": "Sin predicciones aun"}

    precios = []
    alertas = 0

    with open(LOG_PATH, "r") as f:
        reader = csv.DictReader(f)
        for fila in reader:
            try:
                precios.append(float(fila["precio_predicho"]))
                if fila["alerta"]:
                    alertas += 1
            except (ValueError, KeyError):
                continue

    if not precios:
        return {"total_predicciones": 0, "mensaje": "Sin predicciones aun"}

    return {
        "total_predicciones": len(precios),
        "precio_promedio":    round(sum(precios) / len(precios), 2),
        "precio_minimo":      round(min(precios), 2),
        "precio_maximo":      round(max(precios), 2),
        "total_alertas":      alertas,
        "estado":             "OK" if alertas == 0 else "REVISAR",
    }
