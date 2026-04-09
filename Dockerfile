# ─────────────────────────────────────────────
#  Dockerfile — Housing Price API
#  Imagen base: Python 3.11 slim (liviana)
# ─────────────────────────────────────────────

FROM python:3.11-slim

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar dependencias primero (aprovecha cache de Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente
COPY api.py .
COPY config.yaml .
COPY src/ ./src/

# Copiar el modelo entrenado
# Nota: debes entrenar primero con python pipeline.py
COPY modelo_final.pkl .

# Puerto que expone la API
EXPOSE 8000

# Comando para arrancar la API
CMD ["python", "api.py"]
