
"""API REST para predicción de demanda del sistema MiBici."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from mibici.api.model_service import DemandModelService
from mibici.api.schemas import (
    DemandFeatures,
    DemandPrediction,
    HealthResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo al iniciar la aplicación."""
    app.state.demand_service = DemandModelService.load()
    yield


app = FastAPI(
    title="MiBici Demand Prediction API",
    description=(
        "API REST para predecir la demanda de viajes del día siguiente "
        "utilizando el modelo P1 registrado en MLflow."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    """Comprueba que el servicio tiene un modelo cargado."""
    service = request.app.state.demand_service

    return HealthResponse(
        status="ok",
        model_name=service.model_name,
        model_version=service.model_version,
    )


@app.post(
    "/predict/demand",
    response_model=DemandPrediction,
)
def predict_demand(features: DemandFeatures, request: Request):
    """Recibe características de una estación y predice demanda."""
    service = request.app.state.demand_service
    return service.predict(features)
