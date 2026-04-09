"""
src/monitor.py — Monitoreo y gestión de datos de producción.

Archivos que maneja:
  predictions_log.csv  → inputs de cada predicción (sin precio real)
  training_data.csv    → inputs + precio real confirmado por el usuario
"""

import csv
import os
from datetime import datetime


PREDICTIONS_LOG = "predictions_log.csv"
TRAINING_DATA   = "training_data.csv"

# Columnas de inputs
FEATURE_COLUMNS = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM",
    "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"
]

# Rango normal del precio predicho
PRECIO_MIN = 5.0
PRECIO_MAX = 50.0


def inicializar_logs():
    """Crea los archivos de log si no existen."""
    # Log de predicciones — sin precio real
    if not os.path.exists(PREDICTIONS_LOG):
        with open(PREDICTIONS_LOG, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "timestamp"] + FEATURE_COLUMNS + ["precio_predicho", "alerta"])
        print(f"Log de predicciones creado: {PREDICTIONS_LOG}")

    # Datos de entrenamiento — con precio real confirmado
    if not os.path.exists(TRAINING_DATA):
        with open(TRAINING_DATA, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(FEATURE_COLUMNS + ["MEDV"])
        print(f"Archivo de entrenamiento creado: {TRAINING_DATA}")


def registrar_prediccion(features: dict, precio: float) -> str:
    """
    Registra una predicción en el log.
    Retorna el ID de la predicción para referencia futura.
    """
    prediction_id = datetime.now().strftime("%Y%m%d%H%M%S%f")

    alerta = ""
    if precio < PRECIO_MIN:
        alerta = f"PRECIO_BAJO ({precio:.2f} < {PRECIO_MIN})"
    elif precio > PRECIO_MAX:
        alerta = f"PRECIO_ALTO ({precio:.2f} > {PRECIO_MAX})"

    fila = [prediction_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    fila += [features[col] for col in FEATURE_COLUMNS]
    fila += [precio, alerta]

    with open(PREDICTIONS_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fila)

    if alerta:
        print(f"  ALERTA: {alerta}")

    return prediction_id


def guardar_feedback(precios_reales: list[float]) -> dict:
    """
    Toma las últimas N predicciones sin feedback y les asigna
    los precios reales proporcionados por el usuario.

    precios_reales: lista de precios en el mismo orden que las predicciones
                   pendientes de feedback.

    Retorna un resumen de cuántos registros se guardaron.
    """
    # Leer predicciones pendientes (las que no tienen precio real aún)
    predicciones_pendientes = _leer_predicciones_pendientes()

    if len(precios_reales) != len(predicciones_pendientes):
        raise ValueError(
            f"Cantidad incorrecta de precios. "
            f"Hay {len(predicciones_pendientes)} predicciones pendientes "
            f"pero enviaste {len(precios_reales)} precios reales."
        )

    # Guardar inputs + precio real en training_data.csv
    registros_guardados = 0
    with open(TRAINING_DATA, "a", newline="") as f:
        writer = csv.writer(f)
        for prediccion, precio_real in zip(predicciones_pendientes, precios_reales):
            fila = [prediccion[col] for col in FEATURE_COLUMNS] + [precio_real]
            writer.writerow(fila)
            registros_guardados += 1

    # Marcar predicciones como procesadas
    _marcar_como_procesadas(len(precios_reales))

    return {
        "registros_guardados": registros_guardados,
        "total_datos_entrenamiento": _contar_filas(TRAINING_DATA),
    }


def obtener_predicciones_pendientes() -> dict:
    """
    Retorna las predicciones que aún no tienen precio real asignado.
    El usuario las usa para saber qué precios reales debe ingresar.
    """
    pendientes = _leer_predicciones_pendientes()

    return {
        "total_pendientes": len(pendientes),
        "predicciones": [
            {
                "id":              p["id"],
                "timestamp":       p["timestamp"],
                "precio_predicho": p["precio_predicho"],
                "features":        {col: p[col] for col in FEATURE_COLUMNS},
            }
            for p in pendientes
        ],
    }


def obtener_resumen() -> dict:
    """Resumen general de predicciones y datos de entrenamiento."""
    if not os.path.exists(PREDICTIONS_LOG):
        return {"total_predicciones": 0, "mensaje": "Sin predicciones aun"}

    precios    = []
    alertas    = 0
    pendientes = 0

    with open(PREDICTIONS_LOG, "r") as f:
        reader = csv.DictReader(f)
        for fila in reader:
            try:
                precios.append(float(fila["precio_predicho"]))
                if fila["alerta"]:
                    alertas += 1
                if fila.get("procesado", "") != "si":
                    pendientes += 1
            except (ValueError, KeyError):
                continue

    if not precios:
        return {"total_predicciones": 0, "mensaje": "Sin predicciones aun"}

    return {
        "total_predicciones":       len(precios),
        "precio_promedio":          round(sum(precios) / len(precios), 2),
        "precio_minimo":            round(min(precios), 2),
        "precio_maximo":            round(max(precios), 2),
        "total_alertas":            alertas,
        "pendientes_de_feedback":   pendientes,
        "datos_para_reentrenar":    _contar_filas(TRAINING_DATA),
        "estado":                   "OK" if alertas == 0 else "REVISAR",
    }


# ── Funciones internas ────────────────────────────────────────────────────────

def _leer_predicciones_pendientes() -> list:
    """Lee las predicciones que no han sido procesadas aún."""
    if not os.path.exists(PREDICTIONS_LOG):
        return []

    pendientes = []
    with open(PREDICTIONS_LOG, "r") as f:
        reader = csv.DictReader(f)
        for fila in reader:
            if fila.get("procesado", "") != "si":
                pendientes.append(fila)
    return pendientes


def _marcar_como_procesadas(n: int):
    """Marca las primeras N predicciones pendientes como procesadas."""
    if not os.path.exists(PREDICTIONS_LOG):
        return

    filas = []
    with open(PREDICTIONS_LOG, "r") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        # Agregar columna procesado si no existe
        if "procesado" not in fieldnames:
            fieldnames = fieldnames + ["procesado"]

        procesadas = 0
        for fila in reader:
            if fila.get("procesado", "") != "si" and procesadas < n:
                fila["procesado"] = "si"
                procesadas += 1
            filas.append(fila)

    with open(PREDICTIONS_LOG, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filas)


def _contar_filas(path: str) -> int:
    """Cuenta las filas de datos en un CSV (sin el header)."""
    if not os.path.exists(path):
        return 0
    with open(path, "r") as f:
        return sum(1 for _ in f) - 1
