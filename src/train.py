"""
src/train.py — Entrenamiento del modelo lineal con MLflow.
"""

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def entrenar(X_train, X_test, y_train, y_test, version, responsable):
    """
    Entrena un modelo de regresión lineal con todas las features.
    Registra parámetros, métricas y el modelo en MLflow.

    Retorna el modelo entrenado.
    """
    print("  Entrenando modelo de regresión lineal...")

    with mlflow.start_run(run_name=f"LinearRegression_{version}") as run:

        # Tags — quién, cuándo, qué versión
        mlflow.set_tag("version",     version)
        mlflow.set_tag("responsable", responsable)
        mlflow.set_tag("modelo",      "LinearRegression")
        mlflow.set_tag("dataset",     "Boston Housing")

        # Registrar parámetros del modelo
        mlflow.log_param("n_features",      X_train.shape[1])
        mlflow.log_param("n_train_samples", X_train.shape[0])
        mlflow.log_param("n_test_samples",  X_test.shape[0])
        mlflow.log_param("scaler",          "StandardScaler")

        # Pipeline: escalar + regresión lineal
        modelo = Pipeline([
            ("scaler",    StandardScaler()),
            ("regresion", LinearRegression()),
        ])

        # Entrenar
        modelo.fit(X_train, y_train)

        # Predecir y calcular métricas
        y_pred = modelo.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2   = float(r2_score(y_test, y_pred))
        mae  = float(np.mean(np.abs(y_test - y_pred)))

        # Registrar métricas en MLflow
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2",   r2)
        mlflow.log_metric("mae",  mae)

        print(f"  RMSE : {rmse:.4f}")
        print(f"  R2   : {r2:.4f}")
        print(f"  MAE  : {mae:.4f}")

        # Guardar modelo en MLflow
        mlflow.sklearn.log_model(
            sk_model=modelo,
            artifact_path="modelo",
            registered_model_name="housing_linear_model",
        )

        run_id = run.info.run_id
        print(f"  run_id: {run_id}")

    return modelo, run_id, rmse, r2, mae
