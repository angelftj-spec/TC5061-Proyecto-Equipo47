
"""Esquemas de entrada y salida para la API MiBici."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DemandFeatures(BaseModel):
    """Características requeridas por el modelo P1."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )

    # Actividad reciente de la estación.
    departures: int = Field(ge=0)
    arrivals: int = Field(ge=0)

    # Información geográfica.
    latitud: float | None = Field(ge=-90, le=90)
    longitud: float | None = Field(ge=-180, le=180)
    capacidad: float | None = Field(ge=0)

    # Variables de calendario.
    day_of_week: int = Field(ge=0, le=6)
    day_of_month: int = Field(ge=1, le=31)
    month: int = Field(ge=1, le=12)
    is_weekend: Literal[0, 1]

    # Codificación cíclica del calendario.
    day_of_week_sin: float = Field(ge=-1, le=1)
    day_of_week_cos: float = Field(ge=-1, le=1)
    month_sin: float = Field(ge=-1, le=1)
    month_cos: float = Field(ge=-1, le=1)

    # Variables históricas de salidas.
    departures_lag_1: float = Field(ge=0)
    departures_lag_7: float = Field(ge=0)
    departures_mean_prev_7d: float = Field(ge=0)
    departures_mean_prev_28d: float = Field(ge=0)

    # Variables históricas de llegadas.
    arrivals_lag_1: float = Field(ge=0)
    arrivals_lag_7: float = Field(ge=0)
    arrivals_mean_prev_7d: float = Field(ge=0)
    arrivals_mean_prev_28d: float = Field(ge=0)


class DemandPrediction(BaseModel):
    """Respuesta de la API para predicción de demanda."""

    model_config = ConfigDict(allow_inf_nan=False)

    predicted_trips_next_day: float = Field(ge=0)
    model_name: str
    model_version: str


class HealthResponse(BaseModel):
    """Estado del servicio."""

    status: str
    model_name: str
    model_version: str