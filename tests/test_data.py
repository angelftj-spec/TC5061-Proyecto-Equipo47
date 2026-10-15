
"""Pruebas básicas de normalización e ingesta de MiBici."""

import pandas as pd

from mibici.dataset import (
    STATION_COLUMN_ALIASES,
    build_datetime,
    clean_stations,
    iter_months,
    normalize_columns,
    normalize_column_name,
)


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