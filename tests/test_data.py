import numpy as np
import pandas as pd
import pytest

from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from mibici.modeling import run_experiments as exp
from mibici.dataset import (
    STATION_COLUMN_ALIASES,
    build_datetime,
    clean_stations,
    iter_months,
    normalize_columns,
    normalize_column_name,
)


# --------------------------------------------------
# 1. Pruebas básicas de normalización e ingesta de MiBici.
# --------------------------------------------------

def test_normalize_column_name():
    """Los nombres deben normalizarse a snake_case sin acentos."""
    assert normalize_column_name("Inicio del viaje") == "inicio_del_viaje"
    assert normalize_column_name("Niños Héroes") == "ninos_heroes"


def test_station_column_aliases():
    """La nomenclatura debe normalizarse a nombres canónicos."""
    raw = pd.DataFrame(
        {
            "id": [2],
            "name": ["Estación de prueba"],
            "latitude": [20.66],
            "longitude": [-103.34],
            "dpcapacity": [15],
        }
    )

    result = normalize_columns(raw, STATION_COLUMN_ALIASES)

    assert "estacion_id" in result.columns
    assert "nombre_estacion" in result.columns
    assert "latitud" in result.columns
    assert "longitud" in result.columns
    assert "capacidad" in result.columns


def test_build_datetime_supports_iso_and_day_first_dates():
    """Las fechas ISO y DD/MM/YYYY deben interpretarse correctamente."""
    frame = pd.DataFrame(
        {
            "inicio_viaje": [
                "01/11/2023 0:01",
                "2024-03-01 00:00:22",
                "01/09/2024 00:00",
                "01/10/2024 00:00",
            ]
        }
    )

    result = build_datetime(
        frame,
        "inicio_viaje",
        "fecha_inicio",
        "hora_inicio",
    )

    expected = [
        "2023-11-01 00:01:00",
        "2024-03-01 00:00:22",
        "2024-09-01 00:00:00",
        "2024-10-01 00:00:00",
    ]

    assert result.dt.strftime("%Y-%m-%d %H:%M:%S").tolist() == expected


def test_iter_months_includes_both_endpoints():
    """El rango mensual debe incluir el primer mes y el último."""
    assert iter_months("2023-11", "2024-02") == [
        "2023-11",
        "2023-12",
        "2024-01",
        "2024-02",
    ]


def test_clean_stations_normalizes_capacity_and_ids():
    """La limpieza debe producir IDs y capacidades utilizables."""
    raw = pd.DataFrame(
        {
            "id": [2, 3],
            "name": ["Estación A", "Estación B"],
            "latitude": [20.66, 20.67],
            "longitude": [-103.34, -103.35],
            "dpcapacity": [15, 0],
        }
    )

    result = clean_stations(raw)

    assert result["estacion_id"].tolist() == ["2", "3"]
    assert result["capacidad"].tolist() == [15, 0]
    assert result["capacidad"].notna().all()


# --------------------------------------------------
# 2. Pruebas de procesamiento y división temporal
# --------------------------------------------------

def test_temporal_split_filtra_fechas_del_periodo():
    """Solo debe conservar las fechas dentro del intervalo indicado."""
    frame = pd.DataFrame({
        "target_date": [
            "2025-01-01",
            "2025-01-15",
            "2025-02-01",
            "fecha_invalida",
        ],
        "station_id": [1, 2, 3, 4],
    })

    result = exp.temporal_split(frame, "2025-01", "2025-01")

    assert len(result) == 2
    assert result["station_id"].tolist() == [1, 2]
    assert result is not frame


def test_temporal_split_falla_si_falta_target_date():
    """Debe rechazar un DataFrame sin la columna de fecha requerida."""
    frame = pd.DataFrame({"station_id": [1, 2]})

    with pytest.raises(ValueError, match="target_date"):
        exp.temporal_split(frame, "2025-01", "2025-02")


def test_get_supervised_splits_rechaza_conjunto_vacio(monkeypatch):
    """Debe fallar cuando alguno de los periodos no contiene registros."""
    monkeypatch.setattr(exp, "TRAIN_START_MONTH", "2025-01")
    monkeypatch.setattr(exp, "TRAIN_END_MONTH", "2025-01")
    monkeypatch.setattr(exp, "VALIDATION_START_MONTH", "2025-02")
    monkeypatch.setattr(exp, "VALIDATION_END_MONTH", "2025-02")
    monkeypatch.setattr(exp, "TEST_START_MONTH", "2025-03")
    monkeypatch.setattr(exp, "TEST_END_MONTH", "2025-03")

    frame = pd.DataFrame({
        "target_date": ["2025-01-10", "2025-03-10"],
        "station_id": [1, 2],
    })

    with pytest.raises(ValueError, match="validation está vacío"):
        exp.get_supervised_splits(frame)


# --------------------------------------------------
# 3. Pruebas de utilidades y métricas
# --------------------------------------------------

def test_placeholder_x_genera_matriz_correcta():
    """Debe generar una columna ficticia de ceros."""
    result = exp.placeholder_x(4)

    assert result.shape == (4, 1)
    assert result.dtype == np.float64
    assert np.all(result == 0)


def test_placeholder_x_admite_cero_registros():
    """Una entrada de tamaño cero debe producir una matriz vacía válida."""
    result = exp.placeholder_x(0)

    assert result.shape == (0, 1)


def test_placeholder_x_rechaza_tamano_negativo():
    """No se deben aceptar dimensiones negativas."""
    with pytest.raises(ValueError):
        exp.placeholder_x(-1)


def test_classification_metrics_calcula_metricas_correctas():
    """Comprueba las métricas con predicciones conocidas."""
    y_true = pd.Series([0, 0, 1, 1])
    predictions = np.array([0, 0, 1, 0])

    metrics = exp.classification_metrics(
        y_true, predictions, prefix="test"
    )

    assert metrics["test_accuracy"] == pytest.approx(0.75)
    assert metrics["test_balanced_accuracy"] == pytest.approx(0.75)
    assert metrics["test_precision"] == pytest.approx(1.0)
    assert metrics["test_recall"] == pytest.approx(0.5)
    assert metrics["test_f1"] == pytest.approx(2 / 3)


def test_classification_metrics_maneja_predicciones_sin_positivos():
    """La precisión, el recall y el F1 deben ser cero si no hay positivos predichos."""
    y_true = pd.Series([0, 1, 1, 0])
    predictions = np.array([0, 0, 0, 0])

    metrics = exp.classification_metrics(
        y_true, predictions, prefix="test"
    )

    assert metrics["test_precision"] == 0.0
    assert metrics["test_recall"] == 0.0
    assert metrics["test_f1"] == 0.0


# --------------------------------------------------
# 4. Prueba de reproducibilidad del entrenamiento
# --------------------------------------------------

def entrenar_kmeans(datos, semilla):
    """Replica el preprocesamiento y entrenamiento utilizados en P3."""
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "kmeans",
            KMeans(
                n_clusters=2,
                random_state=semilla,
                n_init=20,
            ),
        ),
    ])

    pipeline.fit(datos)
    labels = pipeline.named_steps["kmeans"].labels_
    datos_escalados = pipeline[:-1].transform(datos)

    silhouette = silhouette_score(datos_escalados, labels)
    inertia = pipeline.named_steps["kmeans"].inertia_

    return labels, silhouette, inertia


def test_kmeans_es_reproducible_con_la_misma_semilla():
    """Dos entrenamientos equivalentes deben generar los mismos resultados."""
    datos = pd.DataFrame({
        "viajes_promedio": [
            10, 11, 9, 12, 10.5, 100, 105, 98, 110, 102
        ],
        "capacidad": [
            20, 22, 19, 21, 20, 150, 155, 148, 160, 152
        ],
    })

    semilla = exp.RANDOM_SEED

    labels_1, silhouette_1, inertia_1 = entrenar_kmeans(
        datos, semilla
    )
    labels_2, silhouette_2, inertia_2 = entrenar_kmeans(
        datos, semilla
    )

    np.testing.assert_array_equal(labels_1, labels_2)
    assert silhouette_1 == pytest.approx(silhouette_2)
    assert inertia_1 == pytest.approx(inertia_2)
