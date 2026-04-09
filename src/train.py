"""
src/train.py — Entrenamiento con escalado selectivo y quality gate.

CHAS es variable categórica (0 o 1) — no se escala.
Todas las demás features numéricas sí se escalan con StandardScaler.

Quality gate:
  - Si existe un modelo en producción, solo lo reemplaza si el nuevo es mejor.
  - Compara por RMSE sobre el test set.

Para agregar un nuevo modelo:
  1. Importarlo arriba
  2. Agregarlo al diccionario 'candidatos'
  3. Correr: python pipeline.py --version vX.X
"""

import joblib
import os
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES_CATEGORICAS = ["CHAS"]
MODEL_PATH           = "modelo_final.pkl"


def _construir_preprocesador(X_train: pd.DataFrame) -> ColumnTransformer:
    """Escala todas las columnas menos CHAS."""
    features_numericas = [c for c in X_train.columns if c not in FEATURES_CATEGORICAS]
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), features_numericas),
        ("cat", "passthrough",    FEATURES_CATEGORICAS),
    ])


def _rmse_modelo_actual(X_test, y_test) -> float:
    """
    Calcula el RMSE del modelo actualmente en producción.
    Retorna infinito si no hay modelo guardado.
    """
    if not os.path.exists(MODEL_PATH):
        return float("inf")

    try:
        modelo_actual = joblib.load(MODEL_PATH)
        y_pred        = modelo_actual.predict(X_test)
        return float(np.sqrt(mean_squared_error(y_test, y_pred)))
    except Exception:
        return float("inf")


def _entrenar_candidato(nombre, estimador, X_train, X_test, y_train, y_test):
    """Entrena un candidato como run hijo en MLflow."""
    with mlflow.start_run(run_name=nombre, nested=True):
        mlflow.set_tag("modelo", nombre)

        modelo = Pipeline([
            ("preprocesador", _construir_preprocesador(X_train)),
            ("regresion",     estimador),
        ])

        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2   = float(r2_score(y_test, y_pred))
        mae  = float(np.mean(np.abs(y_test - y_pred)))

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2",   r2)
        mlflow.log_metric("mae",  mae)
        mlflow.sklearn.log_model(modelo, "modelo")

        print(f"  {nombre:<20} RMSE: {rmse:.4f} | R2: {r2:.4f} | MAE: {mae:.4f}")

    return modelo, rmse, r2, mae


def entrenar(X_train, X_test, y_train, y_test, version, responsable):
    """
    Entrena todos los candidatos, aplica quality gate y
    solo reemplaza el modelo en producción si el nuevo es mejor.
    """

    # ── Candidatos ────────────────────────────────────────────────────────────
    candidatos = {
        "LinearRegression": LinearRegression(),
        "Ridge_alpha1":     Ridge(alpha=1.0),
    }
    # ─────────────────────────────────────────────────────────────────────────

    mejor_modelo = None
    mejor_rmse   = float("inf")
    mejor_nombre = ""
    mejor_r2     = 0.0
    mejor_mae    = 0.0

    features_numericas = [c for c in X_train.columns if c not in FEATURES_CATEGORICAS]

    # RMSE del modelo actual en producción (para el quality gate)
    rmse_produccion = _rmse_modelo_actual(X_test, y_test)
    hay_modelo      = rmse_produccion != float("inf")

    if hay_modelo:
        print(f"\n  Modelo actual en producción — RMSE: {rmse_produccion:.4f}")
    else:
        print("\n  Sin modelo en producción — se guardará el ganador directamente.")

    with mlflow.start_run(run_name=f"Experimento_{version}") as run:

        mlflow.set_tag("version",              version)
        mlflow.set_tag("responsable",          responsable)
        mlflow.set_tag("dataset",              "Boston Housing")
        mlflow.log_param("n_candidatos",       len(candidatos))
        mlflow.log_param("n_train_samples",    X_train.shape[0])
        mlflow.log_param("n_features_total",   X_train.shape[1])
        mlflow.log_param("features_sin_escalar", str(FEATURES_CATEGORICAS))

        if hay_modelo:
            mlflow.log_metric("rmse_modelo_produccion", rmse_produccion)

        run_id = run.info.run_id

        print(f"\n  Evaluando {len(candidatos)} modelos...")
        print(f"  Features escaladas  : {features_numericas}")
        print(f"  Features sin escalar: {FEATURES_CATEGORICAS}")

        for nombre, estimador in candidatos.items():
            modelo, rmse, r2, mae = _entrenar_candidato(
                nombre, estimador,
                X_train, X_test, y_train, y_test,
            )
            if rmse < mejor_rmse:
                mejor_rmse   = rmse
                mejor_modelo = modelo
                mejor_nombre = nombre
                mejor_r2     = r2
                mejor_mae    = mae

        mlflow.set_tag("mejor_candidato", mejor_nombre)
        mlflow.log_metric("mejor_rmse_candidato", mejor_rmse)
        mlflow.log_metric("mejor_r2",             mejor_r2)

        # ── Quality gate ──────────────────────────────────────────────────────
        if mejor_rmse < rmse_produccion:
            # El nuevo modelo es mejor — reemplazar
            joblib.dump(mejor_modelo, MODEL_PATH)
            mlflow.set_tag("resultado_quality_gate", "APROBADO — modelo reemplazado")
            mlflow.log_metric("mejora_rmse", rmse_produccion - mejor_rmse)

            print(f"\n  Quality gate: APROBADO")
            print(f"  RMSE anterior : {rmse_produccion:.4f}")
            print(f"  RMSE nuevo    : {mejor_rmse:.4f}")
            print(f"  Mejora        : {rmse_produccion - mejor_rmse:.4f}")
            print(f"  Modelo guardado: {MODEL_PATH}")
            modelo_guardado = True
        else:
            # El modelo actual es mejor — no reemplazar
            mlflow.set_tag("resultado_quality_gate", "RECHAZADO — modelo actual es mejor")

            print(f"\n  Quality gate: RECHAZADO")
            print(f"  RMSE actual  : {rmse_produccion:.4f}")
            print(f"  RMSE nuevo   : {mejor_rmse:.4f}")
            print(f"  El modelo en produccion sigue siendo el mejor.")
            modelo_guardado = False

    return mejor_modelo, run_id, mejor_rmse, mejor_r2, mejor_mae, modelo_guardado
