
"""Pruebas automatizadas de la API REST de MiBici."""

import pytest
from fastapi.testclient import TestClient

from mibici.api.main import app
from mibici.api.model_service import DemandModelService
from mibici.modeling.predict import P1_FEATURES, P1_MODEL_NAME


@pytest.fixture
def valid_payload():
    """Entrada válida con las 21 características del modelo P1."""
    return {
        "departures": 20,
        "arrivals": 17,
        "latitud": 20.674,
        "longitud": -103.35,
        "capacidad": 25,
        "day_of_week": 2,
        "day_of_month": 18,
        "month": 9,
        "is_weekend": 0,
        "day_of_week_sin": 0.9749,
        "day_of_week_cos": -0.2225,
        "month_sin": -0.866,
        "month_cos": -0.5,
        "departures_lag_1": 19,
        "departures_lag_7": 22,
        "departures_mean_prev_7d": 18.5,
        "departures_mean_prev_28d": 20.1,
        "arrivals_lag_1": 16,
        "arrivals_lag_7": 19,
        "arrivals_mean_prev_7d": 17.3,
        "arrivals_mean_prev_28d": 18.2,
    }


@pytest.fixture
def client(monkeypatch):
    """Inicia FastAPI usando un modelo simulado."""

    class FakeModel:
        def predict(self, frame):
            # Validamos que el DataFrame conserve el esquema.
            assert frame.columns.tolist() == P1_FEATURES
            assert frame.shape == (1, len(P1_FEATURES))
            return [23.5]

    def fake_load(cls):
        return cls(FakeModel(), version="test")

    monkeypatch.setattr(
        DemandModelService,
        "load",
        classmethod(fake_load),
    )

    with TestClient(app) as test_client:
        yield test_client


def test_health_returns_ok(client):
    """El servicio debe informar que tiene un modelo cargado."""
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["model_name"] == P1_MODEL_NAME
    assert data["model_version"] == "test"


def test_predict_valid_input(client, valid_payload):
    """Una petición válida debe producir una predicción JSON."""
    response = client.post("/predict/demand", json=valid_payload)

    assert response.status_code == 200

    data = response.json()
    assert data["predicted_trips_next_day"] == 23.5
    assert data["model_name"] == P1_MODEL_NAME
    assert data["model_version"] == "test"


def test_predict_rejects_negative_departures(client, valid_payload):
    """No deben aceptarse cantidades negativas de viajes."""
    valid_payload["departures"] = -10

    response = client.post("/predict/demand", json=valid_payload)

    assert response.status_code == 422


def test_predict_rejects_missing_feature(client, valid_payload):
    """La API debe rechazar entradas incompletas."""
    del valid_payload["month"]

    response = client.post("/predict/demand", json=valid_payload)

    assert response.status_code == 422


def test_predict_rejects_unknown_field(client, valid_payload):
    """La API debe rechazar características no contempladas."""
    valid_payload["unknown_feature"] = 123

    response = client.post("/predict/demand", json=valid_payload)

    assert response.status_code == 422


def test_predict_accepts_missing_coordinates(client, valid_payload):
    """La API permite coordenadas nulas para imputación."""
    valid_payload["latitud"] = None

    response = client.post("/predict/demand", json=valid_payload)

    assert response.status_code == 200


def test_openapi_documentation(client):
    """Swagger debe publicar los endpoints implementados."""
    response = client.get("/openapi.json")

    assert response.status_code == 200

    paths = response.json()["paths"]
    assert "/health" in paths
    assert "/predict/demand" in paths
