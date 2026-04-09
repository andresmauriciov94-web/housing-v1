"""
pipeline.py — Orquestador del ciclo de entrenamiento.

Uso:
    python pipeline.py
    python pipeline.py --version v1.0
    python pipeline.py --version v2.0 --responsable Juan
"""

import argparse
import yaml
import mlflow

from src.data  import get_raw_data, clean_data
from src.train import entrenar


def cargar_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Pipeline MLOps — Housing Price")
    parser.add_argument("--version",     type=str, default="v1.0")
    parser.add_argument("--responsable", type=str, default="Equipo_MLOps")
    args = parser.parse_args()

    cfg = cargar_config()

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    print("\n" + "=" * 50)
    print(f"  PIPELINE — Housing Price {args.version}")
    print("=" * 50)

    # ── 1. Datos ──────────────────────────────────────────────────────────────
    print("\n[1/2] Obteniendo y limpiando datos...")
    df_raw = get_raw_data()

    # clean_data retorna también la figura de distribución MEDV
    X_train, X_test, y_train, y_test, fig_medv = clean_data(
        df_raw,
        precio_limite = cfg["data"]["precio_limite"],
        test_size     = cfg["data"]["test_size"],
        random_state  = cfg["data"]["random_state"],
    )

    # ── 2. Entrenamiento + quality gate ───────────────────────────────────────
    print("\n[2/2] Entrenando modelos...")
    mlflow.end_run()  # Cerrar cualquier run activo antes de iniciar

    # fig_medv se loggea DENTRO del run padre en train.py
    modelo, run_id, rmse, r2, mae, modelo_guardado = entrenar(
        X_train, X_test, y_train, y_test,
        version     = args.version,
        responsable = args.responsable,
        fig_medv    = fig_medv,
    )

    # ── Resumen ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  PIPELINE COMPLETADO")
    print(f"  Versión         : {args.version}")
    print(f"  run_id          : {run_id}")
    print(f"  RMSE            : {rmse:.4f}")
    print(f"  R2              : {r2:.4f}")
    print(f"  MAE             : {mae:.4f}")
    print(f"  Modelo guardado : {'SI' if modelo_guardado else 'NO — el actual es mejor'}")
    print("=" * 50)
    print("\nVer en MLflow:")
    print("  mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
