"""
deploy.py — Script de despliegue de la API.

Verifica que el modelo existe y levanta la API.

Uso:
    python deploy.py
    python deploy.py --port 8080
"""

import argparse
import os
import sys
import yaml
import uvicorn


def cargar_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def verificar_modelo(ruta: str) -> bool:
    if not os.path.exists(ruta):
        print(f"ERROR: modelo no encontrado en '{ruta}'")
        print("Ejecuta primero: python pipeline.py --version v1.0")
        return False
    print(f"Modelo encontrado: {ruta}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Despliegue — Housing Price API")
    parser.add_argument("--port", type=int, default=8000,
                        help="Puerto (default: 8000)")
    args = parser.parse_args()

    cfg = cargar_config()

    print("\n" + "=" * 50)
    print("  DESPLIEGUE — Housing Price API")
    print("=" * 50)

    if not verificar_modelo(cfg["modelo"]["guardar_en"]):
        sys.exit(1)

    print(f"\nAPI corriendo en: http://127.0.0.1:{args.port}")
    print(f"Documentacion  : http://127.0.0.1:{args.port}/docs")
    print("Ctrl+C para detener\n")

    uvicorn.run("api:app", host="127.0.0.1", port=args.port, reload=False)


if __name__ == "__main__":
    main()
