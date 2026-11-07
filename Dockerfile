
FROM python:3.11.9-slim-bookworm

# Configuración de Python y del modelo.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MIBICI_MODEL_PATH=/app/models/p1_serving \
    MIBICI_MODEL_VERSION=1

WORKDIR /app

# Librería del sistema para operaciones numéricas.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias antes de copiar el código.
COPY requirements-api.txt .
RUN python -m pip install -r requirements-api.txt

# Copiar la aplicación y el modelo exportado.
COPY mibici/ ./mibici/
COPY models/p1_serving/ ./models/p1_serving/

# Ejecutar con usuario no privilegiado.
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000

# Iniciar FastAPI.
CMD ["uvicorn", "mibici.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
