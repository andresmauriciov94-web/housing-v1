"""
src/data.py — Obtener y limpiar los datos.

Al reentrenar combina:
  - boston_housing.db   → dataset original
  - training_data.csv   → datos de producción confirmados por el usuario
"""

import os
import sqlite3
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


DB_PATH       = "boston_housing.db"
TRAINING_DATA = "training_data.csv"


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

    print(f"  Datos originales: {len(df)} registros")
    return df


def combinar_con_produccion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Si existen datos de producción confirmados los combina
    con el dataset original.
    """
    if not os.path.exists(TRAINING_DATA):
        return df

    try:
        df_produccion = pd.read_csv(TRAINING_DATA)

        if len(df_produccion) == 0:
            return df

        # Asegurar que las columnas coincidan
        df_produccion.columns = [c.upper() for c in df_produccion.columns]

        df_combinado = pd.concat([df, df_produccion], ignore_index=True)
        print(f"  Datos de producción agregados: {len(df_produccion)} registros")
        print(f"  Total combinado: {len(df_combinado)} registros")
        return df_combinado

    except Exception as e:
        print(f"  Advertencia: no se pudieron cargar datos de producción ({e})")
        return df


def clean_data(df, precio_limite, test_size, random_state):
    """
    Limpia y divide los datos.
    Combina automáticamente con datos de producción si existen.
    """
    # Combinar con datos de producción confirmados
    df = combinar_con_produccion(df)

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
