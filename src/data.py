"""
src/data.py — Obtener y limpiar los datos.

Retorna la figura de distribución MEDV para que pipeline.py
la loggee dentro del run padre de MLflow.
"""

import os
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


DB_PATH       = "boston_housing.db"
TRAINING_DATA = "training_data.csv"


def get_raw_data() -> pd.DataFrame:
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


def _grafica_distribucion_medv(df: pd.DataFrame, precio_limite: float):
    """
    Genera gráfica con histograma y boxplot de MEDV.
    Retorna la figura para que sea loggeada desde pipeline.py.
    """
    n_eliminados = (df["MEDV"] >= precio_limite).sum()
    n_total      = len(df)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Distribución de MEDV — Justificación del corte en {precio_limite}k USD\n"
        f"({n_eliminados} registros eliminados de {n_total} — "
        f"{n_eliminados/n_total*100:.1f}%)",
        fontsize=12, fontweight="bold",
    )

    # ── Histograma ────────────────────────────────────────────────────────────
    ax1.hist(
        df["MEDV"],
        bins=30, color="#4299e1", edgecolor="white",
        alpha=0.8, label="Registros válidos",
    )
    ax1.hist(
        df[df["MEDV"] >= precio_limite]["MEDV"],
        bins=10, color="#e53e3e", edgecolor="white",
        alpha=0.8, label=f"Eliminados (MEDV >= {precio_limite})",
    )
    ax1.axvline(x=precio_limite, color="#e53e3e", linestyle="--",
                linewidth=2, label=f"Límite = {precio_limite}k")
    ax1.set_title("Histograma de MEDV")
    ax1.set_xlabel("Precio medio (miles USD)")
    ax1.set_ylabel("Frecuencia")
    ax1.legend(fontsize=9)
    ax1.annotate(
        f"Valor censurado\n({n_eliminados} registros)",
        xy=(precio_limite, ax1.get_ylim()[1] * 0.6),
        xytext=(precio_limite - 18, ax1.get_ylim()[1] * 0.75),
        arrowprops=dict(arrowstyle="->", color="#e53e3e"),
        color="#e53e3e", fontsize=9,
    )

    # ── Boxplot ───────────────────────────────────────────────────────────────
    ax2.boxplot(
        df["MEDV"], vert=True, patch_artist=True,
        boxprops=dict(facecolor="#bee3f8", color="#2b6cb0"),
        medianprops=dict(color="#2b6cb0", linewidth=2),
        whiskerprops=dict(color="#4299e1"),
        capprops=dict(color="#4299e1"),
        flierprops=dict(marker="o", markerfacecolor="#e53e3e",
                        markersize=5, alpha=0.6),
    )
    ax2.axhline(y=precio_limite, color="#e53e3e", linestyle="--",
                linewidth=2, label=f"Límite = {precio_limite}k")
    ax2.set_title("Boxplot de MEDV")
    ax2.set_ylabel("Precio medio (miles USD)")
    ax2.legend(fontsize=9)

    stats = df["MEDV"].describe()
    stats_text = (
        f"Media:   {stats['mean']:.1f}k\n"
        f"Mediana: {stats['50%']:.1f}k\n"
        f"Std:     {stats['std']:.1f}k\n"
        f"Min:     {stats['min']:.1f}k\n"
        f"Max:     {stats['max']:.1f}k"
    )
    ax2.text(
        1.12, 0.5, stats_text, transform=ax2.transAxes,
        fontsize=9, verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="#ebf8ff", alpha=0.8),
    )

    plt.tight_layout()
    return fig  # ← retorna la figura, NO loggea aquí


def combinar_con_produccion(df: pd.DataFrame) -> pd.DataFrame:
    if not os.path.exists(TRAINING_DATA):
        return df
    try:
        df_prod = pd.read_csv(TRAINING_DATA)
        if len(df_prod) == 0:
            return df
        df_prod.columns = [c.upper() for c in df_prod.columns]
        df_combinado = pd.concat([df, df_prod], ignore_index=True)
        print(f"  Datos de producción agregados: {len(df_prod)} registros")
        print(f"  Total combinado: {len(df_combinado)} registros")
        return df_combinado
    except Exception as e:
        print(f"  Advertencia: no se pudieron cargar datos de producción ({e})")
        return df


def clean_data(df, precio_limite, test_size, random_state):
    """
    Limpia y divide los datos.
    Retorna X_train, X_test, y_train, y_test y la figura MEDV.
    """
    df = combinar_con_produccion(df)

    # Generar figura ANTES del corte — retornar para loggear en pipeline.py
    fig_medv = _grafica_distribucion_medv(df, precio_limite)

    df_clean = df[df["MEDV"] < precio_limite].copy()
    print(f"  Eliminados {len(df) - len(df_clean)} registros con MEDV >= {precio_limite}")

    X = df_clean.drop("MEDV", axis=1)
    y = df_clean["MEDV"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    return X_train, X_test, y_train, y_test, fig_medv
