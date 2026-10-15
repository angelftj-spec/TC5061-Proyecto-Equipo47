
"""Funciones reutilizables de feature engineering para el proyecto MiBici."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import typer
from loguru import logger

from mibici.config import (
    DATA_END_MONTH,
    DATA_START_MONTH,
    PROCESSED_DATA_DIR,
    PROCESSED_STATIONS_DIR,
    PROCESSED_TRIPS_DIR,
)

app = typer.Typer(add_completion=False)

# Umbral inicial para las etiquetas aproximadas de riesgo de P2:
# un flujo neto diario equivalente al 20% de la capacidad de la estación.
# No representa un inventario real de bicicletas o anclajes.
DEFAULT_RISK_CAPACITY_FRACTION = 0.20


def _month_range(start_month: str, end_month: str) -> list[str]:
    """Devuelve los meses inclusivos del periodo solicitado."""
    start = pd.Period(start_month, freq="M")
    end = pd.Period(end_month, freq="M")

    if start > end:
        raise ValueError("El mes inicial no puede ser posterior al final.")

    return [str(month) for month in pd.period_range(start, end, freq="M")]


def _trip_paths(start_month: str, end_month: str) -> list[Path]:
    """Localiza y verifica los Parquet mensuales de viajes."""
    paths = []

    for month in _month_range(start_month, end_month):
        year, number = month.split("-")
        path = PROCESSED_TRIPS_DIR / f"viajes_{year}_{number}.parquet"

        if not path.is_file():
            raise FileNotFoundError(
                f"No se encontró el Parquet de {month}: {path}. "
                "Ejecuta primero la ingesta de MiBici."
            )

        paths.append(path)

    return paths


def _load_station_metadata() -> pd.DataFrame:
    """Carga la nomenclatura procesada y normaliza los IDs."""
    files = sorted(PROCESSED_STATIONS_DIR.glob("*.parquet"))

    if len(files) != 1:
        raise FileNotFoundError(
            "Se esperaba exactamente un Parquet de nomenclatura en "
            f"{PROCESSED_STATIONS_DIR}; encontrados: {len(files)}."
        )

    stations = pd.read_parquet(files[0])

    if "estacion_id" not in stations.columns:
        raise ValueError("La nomenclatura no contiene 'estacion_id'.")

    stations = stations.copy()
    stations["station_id"] = (
        stations["estacion_id"].astype("string").str.strip()
    )
    stations = stations.dropna(subset=["station_id"])
    stations = stations.drop_duplicates("station_id", keep="last")
    stations = stations.drop(columns=["estacion_id"])

    for column in ("latitud", "longitud", "capacidad"):
        if column in stations.columns:
            stations[column] = pd.to_numeric(
                stations[column], errors="coerce"
            )

    return stations


def build_daily_station_flows(
    start_month: str = DATA_START_MONTH,
    end_month: str = DATA_END_MONTH,
) -> pd.DataFrame:
    """Calcula viajes iniciados y terminados por estación y día.

    Cada viaje aporta una salida según su fecha de inicio y una llegada
    según su fecha de finalización. Los CSV mensuales se agregan antes
    de concatenarse para reducir el uso de memoria.

    Returns:
        Una fila por estación y día, con salidas, llegadas y metadatos
        disponibles de la nomenclatura.
    """
    paths = _trip_paths(start_month, end_month)
    departure_parts = []
    arrival_parts = []

    for path in paths:
        logger.info(f"Agregando flujos: {path.name}")
        trips = pd.read_parquet(
            path,
            columns=[
                "inicio_viaje",
                "fin_viaje",
                "origen_id",
                "destino_id",
            ],
        )

        start = pd.to_datetime(trips["inicio_viaje"], errors="coerce")
        end = pd.to_datetime(trips["fin_viaje"], errors="coerce")
        origins = trips["origen_id"].astype("string").str.strip()
        destinations = trips["destino_id"].astype("string").str.strip()

        departures = pd.DataFrame(
            {
                "date": start.dt.normalize(),
                "station_id": origins,
            }
        ).dropna(subset=["date", "station_id"])

        arrivals = pd.DataFrame(
            {
                "date": end.dt.normalize(),
                "station_id": destinations,
            }
        ).dropna(subset=["date", "station_id"])

        departure_parts.append(
            departures.groupby(["date", "station_id"], as_index=False)
            .size()
            .rename(columns={"size": "departures"})
        )

        arrival_parts.append(
            arrivals.groupby(["date", "station_id"], as_index=False)
            .size()
            .rename(columns={"size": "arrivals"})
        )

        del trips, departures, arrivals

    departures = pd.concat(departure_parts, ignore_index=True)
    arrivals = pd.concat(arrival_parts, ignore_index=True)

    departures = (
        departures.groupby(["date", "station_id"], as_index=False)
        ["departures"]
        .sum()
    )
    arrivals = (
        arrivals.groupby(["date", "station_id"], as_index=False)
        ["arrivals"]
        .sum()
    )

    observed_flows = departures.merge(
        arrivals,
        on=["date", "station_id"],
        how="outer",
    )

    stations = _load_station_metadata()

    all_station_ids = set(
        observed_flows["station_id"].dropna().astype(str)
    )
    all_station_ids.update(stations["station_id"].dropna().astype(str))
    station_ids = sorted(all_station_ids)

    first_date = pd.Period(start_month, freq="M").start_time
    last_date = pd.Period(end_month, freq="M").end_time.normalize()
    dates = pd.date_range(first_date, last_date, freq="D")

    grid = pd.MultiIndex.from_product(
        [station_ids, dates],
        names=["station_id", "date"],
    ).to_frame(index=False)

    daily = grid.merge(
        observed_flows,
        on=["date", "station_id"],
        how="left",
        validate="one_to_one",
    )

    daily["departures"] = daily["departures"].fillna(0).astype("int32")
    daily["arrivals"] = daily["arrivals"].fillna(0).astype("int32")

    daily = daily.merge(
        stations,
        on="station_id",
        how="left",
        validate="many_to_one",
    )

    daily = daily.sort_values(
        ["station_id", "date"]
    ).reset_index(drop=True)

    return daily


def _add_history_features(
    daily: pd.DataFrame,
    include_28_day_mean: bool = False,
) -> pd.DataFrame:
    """Añade variables temporales y resúmenes exclusivamente históricos.

    Los rezagos usan días anteriores. Las medias móviles aplican shift(1)
    antes del rolling para que el día que se predice no forme parte de
    sus propias variables históricas.
    """
    result = daily.copy()
    result["date"] = pd.to_datetime(result["date"])
    result = result.sort_values(
        ["station_id", "date"]
    ).reset_index(drop=True)

    result["day_of_week"] = result["date"].dt.dayofweek
    result["day_of_month"] = result["date"].dt.day
    result["month"] = result["date"].dt.month
    result["is_weekend"] = (
        result["day_of_week"] >= 5
    ).astype("int8")

    # Codificación cíclica para variables de calendario.
    result["day_of_week_sin"] = np.sin(
        2 * np.pi * result["day_of_week"] / 7
    )
    result["day_of_week_cos"] = np.cos(
        2 * np.pi * result["day_of_week"] / 7
    )
    result["month_sin"] = np.sin(
        2 * np.pi * (result["month"] - 1) / 12
    )
    result["month_cos"] = np.cos(
        2 * np.pi * (result["month"] - 1) / 12
    )

    for column in ("departures", "arrivals"):
        grouped = result.groupby("station_id", sort=False)[column]

        for lag in (1, 7):
            result[f"{column}_lag_{lag}"] = grouped.shift(lag)

        result[f"{column}_mean_prev_7d"] = grouped.transform(
            lambda values: values.shift(1).rolling(
                7, min_periods=7
            ).mean()
        )

        if include_28_day_mean:
            result[f"{column}_mean_prev_28d"] = grouped.transform(
                lambda values: values.shift(1).rolling(
                    28, min_periods=28
                ).mean()
            )

    return result


def build_p1_regression_features(
    daily_flows: pd.DataFrame,
) -> pd.DataFrame:
    """Construye muestras para predecir viajes iniciados al día siguiente.

    La fila representa la información disponible hasta el final del día
    date. El objetivo es el número de salidas en target_date = date + 1.
    """
    features = _add_history_features(
        daily_flows, include_28_day_mean=True
    )

    features["target_date"] = (
        features["date"] + pd.Timedelta(days=1)
    )
    features["target_trips_next_day"] = (
        features.groupby("station_id", sort=False)["departures"]
        .shift(-1)
    )

    required = [
        "departures_lag_1",
        "departures_lag_7",
        "departures_mean_prev_7d",
        "departures_mean_prev_28d",
        "target_trips_next_day",
    ]

    features = features.dropna(subset=required).copy()
    features["target_trips_next_day"] = (
        features["target_trips_next_day"].astype("int32")
    )

    return features.reset_index(drop=True)


def build_p2_classification_features(
    daily_flows: pd.DataFrame,
    risk_capacity_fraction: float = DEFAULT_RISK_CAPACITY_FRACTION,
) -> pd.DataFrame:
    """Construye etiquetas aproximadas de presión de disponibilidad para P2.

    La etiqueta usa el flujo neto del día siguiente comparado con la
    capacidad registrada de la estación. Más salidas que llegadas sugieren
    presión de escasez de bicicletas; más llegadas que salidas sugieren
    presión de ocupación de anclajes.

    No equivale a detectar estaciones realmente vacías o llenas: el dataset
    no proporciona inventarios de bicicletas ni ocupación de anclajes.
    """
    if not 0 < risk_capacity_fraction <= 1:
        raise ValueError(
            "risk_capacity_fraction debe estar en el intervalo (0, 1]."
        )

    features = _add_history_features(daily_flows)
    features["target_date"] = (
        features["date"] + pd.Timedelta(days=1)
    )

    grouped = features.groupby("station_id", sort=False)
    next_departures = grouped["departures"].shift(-1)
    next_arrivals = grouped["arrivals"].shift(-1)

    if "capacidad" not in features.columns:
        raise ValueError(
            "No existe la columna 'capacidad' en la nomenclatura. "
            "Revisa el Parquet de estaciones."
        )

    capacity = pd.to_numeric(features["capacidad"], errors="coerce")
    threshold = capacity * risk_capacity_fraction

    valid = (
        next_departures.notna()
        & next_arrivals.notna()
        & capacity.notna()
        & capacity.gt(0)
    )

    # Etiquetas binarias; se dejan nulas las muestras sin capacidad válida.
    bike_pressure = (
        (next_departures - next_arrivals) >= threshold
    )
    dock_pressure = (
        (next_arrivals - next_departures) >= threshold
    )

    features["target_risk_bikes_next_day"] = (
        bike_pressure.astype("float64").where(valid, np.nan)
    )
    features["target_risk_docks_next_day"] = (
        dock_pressure.astype("float64").where(valid, np.nan)
    )
    features["target_risk_station_next_day"] = (
        (bike_pressure | dock_pressure)
        .astype("float64")
        .where(valid, np.nan)
    )

    features = features.dropna(
        subset=[
            "departures_lag_1",
            "departures_lag_7",
            "arrivals_lag_1",
            "arrivals_lag_7",
            "departures_mean_prev_7d",
            "arrivals_mean_prev_7d",
            "target_risk_station_next_day",
        ]
    ).copy()

    label_columns = [
        "target_risk_bikes_next_day",
        "target_risk_docks_next_day",
        "target_risk_station_next_day",
    ]
    for column in label_columns:
        features[column] = features[column].astype("int8")

    return features.reset_index(drop=True)


def build_p3_clustering_features(
    daily_flows: pd.DataFrame,
    start_month: str = DATA_START_MONTH,
    end_month: str = DATA_END_MONTH,
) -> pd.DataFrame:
    """Construye un perfil de uso por estación para clustering.

    Incluye volumen de viajes, desequilibrio entre llegadas y salidas,
    relación entre fines de semana y días laborales, capacidad, coordenadas
    y proporción de salidas por hora del día.

    Las coordenadas se conservan para interpretar y mapear los grupos; la
    preparación del modelo deberá escalar adecuadamente las variables.
    """
    paths = _trip_paths(start_month, end_month)
    hourly_parts = []

    for path in paths:
        logger.info(f"Perfil horario de estaciones: {path.name}")
        trips = pd.read_parquet(
            path,
            columns=["inicio_viaje", "origen_id"],
        )

        start = pd.to_datetime(
            trips["inicio_viaje"], errors="coerce"
        )
        stations = trips["origen_id"].astype("string").str.strip()

        valid = start.notna() & stations.notna()
        departures = pd.DataFrame(
            {
                "station_id": stations.loc[valid].to_numpy(),
                "hour": start.loc[valid].dt.hour.to_numpy(),
            }
        )

        hourly_parts.append(
            departures.groupby(
                ["station_id", "hour"], as_index=False
            )
            .size()
            .rename(columns={"size": "trips"})
        )

        del trips, departures

    hourly = pd.concat(hourly_parts, ignore_index=True)
    hourly = hourly.groupby(
        ["station_id", "hour"], as_index=False
    )["trips"].sum()

    hourly_wide = hourly.pivot(
        index="station_id",
        columns="hour",
        values="trips",
    ).reindex(columns=range(24), fill_value=0)

    hourly_shares = hourly_wide.div(
        hourly_wide.sum(axis=1).replace(0, np.nan),
        axis=0,
    ).fillna(0)

    hourly_shares.columns = [
        f"departure_share_hour_{hour:02d}"
        for hour in range(24)
    ]
    hourly_shares = hourly_shares.reset_index()

    daily = daily_flows.copy()
    daily["net_inflow"] = daily["arrivals"] - daily["departures"]

    summary = daily.groupby("station_id", as_index=False).agg(
        mean_daily_departures=("departures", "mean"),
        median_daily_departures=("departures", "median"),
        std_daily_departures=("departures", "std"),
        mean_daily_arrivals=("arrivals", "mean"),
        mean_daily_net_inflow=("net_inflow", "mean"),
        active_departure_days=(
            "departures",
            lambda values: int((values > 0).sum()),
        ),
        mean_departures_active_day=(
            "departures",
            lambda values: float(values[values > 0].mean())
            if (values > 0).any()
            else 0.0,
        ),
    )

    daily["is_weekend"] = (
        daily["date"].dt.dayofweek >= 5
    )

    rates = daily.groupby(
        ["station_id", "is_weekend"]
    )["departures"].mean().unstack()

    rates = rates.reindex(columns=[False, True])
    rates.columns = ["weekday_mean_departures", "weekend_mean_departures"]

    summary = summary.merge(
        rates.reset_index(),
        on="station_id",
        how="left",
        validate="one_to_one",
    )

    # Suavizado de 1 viaje/día para evitar cocientes indefinidos
    # en estaciones sin viajes entre semana ni en fin de semana.
    summary["weekend_weekday_departures_ratio"] = (
        (summary["weekend_mean_departures"].fillna(0) + 1.0)
        / (summary["weekday_mean_departures"].fillna(0) + 1.0)
    )

    station_metadata = _load_station_metadata()
    required_coordinates = {"latitud", "longitud"}
    if not required_coordinates.issubset(station_metadata.columns):
        raise ValueError(
            "La nomenclatura necesita latitud y longitud para P3. "
            f"Columnas disponibles: {list(station_metadata.columns)}"
        )

    result = station_metadata.merge(
        summary, on="station_id", how="left", validate="one_to_one"
    )
    result = result.merge(
        hourly_shares, on="station_id", how="left", validate="one_to_one"
    )

    hour_columns = [
        column
        for column in result.columns
        if column.startswith("departure_share_hour_")
    ]
    result[hour_columns] = result[hour_columns].fillna(0)

    # Solo estaciones geolocalizadas pueden formar parte del mapa final.
    result = result.dropna(subset=["latitud", "longitud"]).copy()
    result = result.sort_values("station_id").reset_index(drop=True)

    return result


@app.command()
def main(
    start_month: str = typer.Option(
        DATA_START_MONTH, help="Primer mes de datos (YYYY-MM)."
    ),
    end_month: str = typer.Option(
        DATA_END_MONTH, help="Último mes de datos (YYYY-MM)."
    ),
    risk_capacity_fraction: float = typer.Option(
        DEFAULT_RISK_CAPACITY_FRACTION,
        help="Fracción de capacidad para la etiqueta proxy de riesgo P2.",
    ),
    output_dir: Path = typer.Option(
        PROCESSED_DATA_DIR / "features",
        help="Directorio donde se guardan los conjuntos de variables.",
    ),
) -> None:
    """Genera los datasets reutilizables de P1, P2 y P3."""
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Calculando flujos diarios por estación...")
    daily_flows = build_daily_station_flows(start_month, end_month)

    daily_path = output_dir / "daily_station_flows.parquet"
    daily_flows.to_parquet(daily_path, index=False)

    logger.info("Construyendo variables de regresión P1...")
    p1 = build_p1_regression_features(daily_flows)
    p1_path = output_dir / "p1_regression.parquet"
    p1.to_parquet(p1_path, index=False)

    logger.info("Construyendo variables de clasificación P2...")
    p2 = build_p2_classification_features(
        daily_flows,
        risk_capacity_fraction=risk_capacity_fraction,
    )
    p2_path = output_dir / "p2_classification.parquet"
    p2.to_parquet(p2_path, index=False)

    logger.info("Construyendo perfiles de estaciones P3...")
    p3 = build_p3_clustering_features(
        daily_flows,
        start_month=start_month,
        end_month=end_month,
    )
    p3_path = output_dir / "p3_clustering.parquet"
    p3.to_parquet(p3_path, index=False)

    logger.success(f"Flujos diarios: {daily_path} ({len(daily_flows):,} filas)")
    logger.success(f"P1: {p1_path} ({len(p1):,} filas)")
    logger.success(f"P2: {p2_path} ({len(p2):,} filas)")
    logger.success(f"P3: {p3_path} ({len(p3):,} estaciones)")
    logger.info(
        "La etiqueta P2 es una aproximación basada en flujo neto y capacidad; "
        "no representa inventarios observados de bicicletas o anclajes."
    )


if __name__ == "__main__":
    app()