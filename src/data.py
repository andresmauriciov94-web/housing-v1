"""
src/data.py — Obtener y limpiar los datos.
"""

import os
import sqlite3
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


DB_PATH = "boston_housing.db"


def get_raw_data() -> pd.DataFrame:
    """Carga desde SQLite si existe, si no descarga de OpenML."""
    if os.path.exists(DB_PATH):
        print("  Cargando datos desde SQLite local...")
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM boston_data", conn)
        conn.close()
    else:
        print("  Descargando Boston Housing desde OpenML...")
        boston = fetch_openml(data_id=531, as_frame=True, parser="auto")
        df = boston.frame.dropna()

        if "MEDV" not in df.columns and "target" in df.columns:
            df.rename(columns={"target": "MEDV"}, inplace=True)

        df.columns = [c.upper() for c in df.columns]

        conn = sqlite3.connect(DB_PATH)
        df.to_sql("boston_data", conn, index=False, if_exists="replace")
        conn.close()
        print(f"  Dataset guardado en {DB_PATH}")

    print(f"  Filas: {len(df)} | Columnas: {len(df.columns)}")
    return df


def clean_data(df, precio_limite, test_size, random_state):
    """
    Limpia y divide los datos.
    - Elimina MEDV >= precio_limite (valor censurado)
    - Separa features y target
    - Divide en train / test
    """
    # Eliminar valores censurados
    df_clean = df[df["MEDV"] < precio_limite].copy()
    print(f"  Eliminados {len(df) - len(df_clean)} registros con MEDV >= {precio_limite}")

    # Separar X e y
    X = df_clean.drop("MEDV", axis=1)
    y = df_clean["MEDV"]

    # Dividir
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    return X_train, X_test, y_train, y_test
