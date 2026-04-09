"""
pipeline.py — Orquestador principal del ciclo de vida del modelo.

Uso:
    python pipeline.py
    python pipeline.py --version v1.0
    python pipeline.py --version v1.0 --responsable Juan
"""

import argparse
import joblib
import yaml
import mlflow

from src.data  import get_raw_data, clean_data
from src.train import entrenar


def cargar_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def main():
    # Argumentos
    parser = argparse.ArgumentParser(description="Pipeline MLOps — Housing Price")
    parser.add_argument("--version",     type=str, default="v1.0",
                        help="Version del experimento")
    parser.add_argument("--responsable", type=str, default="Equipo_MLOps",
                        help="Nombre del responsable")
    args = parser.parse_args()

    cfg = cargar_config()

    # Configurar MLflow
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    print("\n" + "=" * 50)
    print(f"  PIPELINE — Housing Price Prediction {args.version}")
    print("=" * 50)

    # ── 1. Datos ──────────────────────────────────────────────────────────────
    print("\n[1/3] Obteniendo y limpiando datos...")
    df_raw = get_raw_data()

    X_train, X_test, y_train, y_test = clean_data(
        df_raw,
        precio_limite = cfg["data"]["precio_limite"],
        test_size     = cfg["data"]["test_size"],
        random_state  = cfg["data"]["random_state"],
    )

    # ── 2. Entrenamiento ──────────────────────────────────────────────────────
    print("\n[2/3] Entrenando modelo...")
    modelo, run_id, rmse, r2, mae = entrenar(
        X_train, X_test, y_train, y_test,
        version     = args.version,
        responsable = args.responsable,
    )

    # ── 3. Guardar modelo localmente ──────────────────────────────────────────
    print("\n[3/3] Guardando modelo...")
    ruta = cfg["modelo"]["guardar_en"]
    joblib.dump(modelo, ruta)
    print(f"  Guardado en: {ruta}")

    # Resumen
    print("\n" + "=" * 50)
    print("  COMPLETADO")
    print(f"  Versión   : {args.version}")
    print(f"  run_id    : {run_id}")
    print(f"  RMSE      : {rmse:.4f}")
    print(f"  R2        : {r2:.4f}")
    print(f"  MAE       : {mae:.4f}")
    print("=" * 50)
    print("\nVer en MLflow:")
    print("  mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
