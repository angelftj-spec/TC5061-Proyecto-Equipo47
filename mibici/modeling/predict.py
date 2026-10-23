from pathlib import Path

import mlflow
import pandas as pd
import typer
from loguru import logger

from mibici.config import (
    MLFLOW_EXPERIMENT_NAME,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
)

app = typer.Typer()

FEATURES_DIR = PROCESSED_DATA_DIR / "features"
OUTPUT_DIR = PROCESSED_DATA_DIR / "predictions"

P1_MODEL_NAME = "mibici-equipo47-p1-demanda"
P2_MODEL_NAME = "mibici-equipo47-p2-riesgo"
P3_MODEL_NAME = "mibici-equipo47-p3-grupos"

P1_FEATURES = [
    "departures",
    "arrivals",
    "latitud",
    "longitud",
    "capacidad",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "departures_lag_1",
    "departures_lag_7",
    "departures_mean_prev_7d",
    "departures_mean_prev_28d",
    "arrivals_lag_1",
    "arrivals_lag_7",
    "arrivals_mean_prev_7d",
    "arrivals_mean_prev_28d",
]

P2_FEATURES = [
    "departures",
    "arrivals",
    "latitud",
    "longitud",
    "capacidad",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "departures_lag_1",
    "departures_lag_7",
    "departures_mean_prev_7d",
    "arrivals_lag_1",
    "arrivals_lag_7",
    "arrivals_mean_prev_7d",
]

P3_FEATURES = [
    "latitud",
    "longitud",
    "capacidad",
    "mean_daily_departures",
    "median_daily_departures",
    "std_daily_departures",
    "mean_daily_arrivals",
    "mean_daily_net_inflow",
    "active_departure_days",
    "mean_departures_active_day",
    "weekday_mean_departures",
    "weekend_mean_departures",
    "weekend_weekday_departures_ratio",
    *[f"departure_share_hour_{hour:02d}" for hour in range(24)],
]


def setup_tracking() -> None:
    """Conecta con la base SQLite de MLflow del proyecto."""
    db_path = PROJ_ROOT / "mlflow.db"
    tracking_uri = f"sqlite:///{db_path.as_posix()}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    logger.info(f"MLflow tracking URI: {tracking_uri}")


def load_champion_model(model_name: str):
    """Resuelve el alias champion y carga el modelo por su ID dinámico."""
    alias_uri = f"models:/{model_name}@champion"

    info = mlflow.models.get_model_info(alias_uri)
    model_id = info.model_id

    if not model_id:
        raise RuntimeError(
            f"No se pudo resolver el model_id de {model_name}@champion."
        )

    logger.info(f"Modelo: {model_name}@champion")
    logger.info(f"Model ID resuelto: {model_id}")

    # En este entorno, cargar por ID evita el error de ruta del alias directo.
    return mlflow.pyfunc.load_model(f"models:/{model_id}")


def validate_features(
    frame: pd.DataFrame,
    feature_names: list[str],
    dataset_name: str,
) -> pd.DataFrame:
    """Comprueba que existan las columnas necesarias para la inferencia."""
    missing = sorted(set(feature_names) - set(frame.columns))

    if missing:
        raise ValueError(
            f"Faltan columnas en {dataset_name}: {missing}"
        )

    return frame.loc[:, feature_names]


@app.command()
def main():
    """Genera predicciones de demanda, riesgo y agrupación de estaciones."""
    setup_tracking()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # P1: demanda de viajes para la fecha objetivo más reciente.
    # ---------------------------------------------------------------
    p1_path = FEATURES_DIR / "p1_regression.parquet"
    logger.info(f"Leyendo {p1_path}")
    p1 = pd.read_parquet(p1_path)

    latest_p1_date = p1["date"].max()
    p1_latest = p1.loc[p1["date"] == latest_p1_date].copy()

    X1 = validate_features(p1_latest, P1_FEATURES, "P1").astype(float)
    model1 = load_champion_model(P1_MODEL_NAME)

    p1_output = p1_latest[
        [
            column
            for column in (
                "station_id",
                "nombre_estacion",
                "date",
                "target_date",
            )
            if column in p1_latest.columns
        ]
    ].copy()

    p1_output["predicted_trips_next_day"] = model1.predict(X1)
    p1_output = p1_output.rename(columns={"target_date": "prediction_date"})

    p1_output_path = OUTPUT_DIR / "p1_demand_predictions.csv"
    p1_output.to_csv(p1_output_path, index=False)
    logger.success(
        f"P1: {len(p1_output)} predicciones guardadas en {p1_output_path}"
    )

    # ---------------------------------------------------------------
    # P2: riesgo de estación para la fecha objetivo más reciente.
    # ---------------------------------------------------------------
    p2_path = FEATURES_DIR / "p2_classification.parquet"
    logger.info(f"Leyendo {p2_path}")
    p2 = pd.read_parquet(p2_path)

    latest_p2_date = p2["date"].max()
    p2_latest = p2.loc[p2["date"] == latest_p2_date].copy()

    X2 = validate_features(p2_latest, P2_FEATURES, "P2").astype(float)
    model2 = load_champion_model(P2_MODEL_NAME)

    p2_output = p2_latest[
        [
            column
            for column in (
                "station_id",
                "nombre_estacion",
                "date",
                "target_date",
            )
            if column in p2_latest.columns
        ]
    ].copy()

    p2_output["predicted_station_risk"] = model2.predict(X2)
    p2_output = p2_output.rename(columns={"target_date": "prediction_date"})

    p2_output_path = OUTPUT_DIR / "p2_risk_predictions.csv"
    p2_output.to_csv(p2_output_path, index=False)
    logger.success(
        f"P2: {len(p2_output)} predicciones guardadas en {p2_output_path}"
    )

    # ---------------------------------------------------------------
    # P3: asignación de cluster para cada estación.
    # ---------------------------------------------------------------
    p3_path = FEATURES_DIR / "p3_clustering.parquet"
    logger.info(f"Leyendo {p3_path}")
    p3 = pd.read_parquet(p3_path)

    X3 = validate_features(p3, P3_FEATURES, "P3")
    model3 = load_champion_model(P3_MODEL_NAME)

    p3_output = p3[
        [
            column
            for column in (
                "station_id",
                "nombre_estacion",
                "latitud",
                "longitud",
                "capacidad",
            )
            if column in p3.columns
        ]
    ].copy()

    p3_output["predicted_cluster"] = model3.predict(X3)

    p3_output_path = OUTPUT_DIR / "p3_station_clusters.csv"
    p3_output.to_csv(p3_output_path, index=False)
    logger.success(
        f"P3: {len(p3_output)} estaciones agrupadas en {p3_output_path}"
    )

    logger.success("Inferencia de los tres modelos completada.")


if __name__ == "__main__":
    app()
