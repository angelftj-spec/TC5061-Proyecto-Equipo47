"""Compara los modelos P1 A y B sobre octubre de 2026."""

from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from mibici.config import PROJ_ROOT
from mibici.modeling.run_final_models import P1_FEATURES, P1_TARGET


MODEL_NAME = "mibici-equipo47-p1-demanda"

MODEL_A_VERSION = "2"
MODEL_B_VERSION = "3"

FEATURES_PATH = (
    PROJ_ROOT / "data/processed/features/p1_regression.parquet"
)

RESULTS_DIR = PROJ_ROOT / "reports"
RESULTS_PATH = RESULTS_DIR / "ab_test_p1_oct2026.csv"

PREDICTIONS_PATH = RESULTS_DIR / "ab_test_p1_oct2026_predictions.csv"


def calculate_metrics(y_true, predictions):
    return {
        "mae": mean_absolute_error(y_true, predictions),
        "rmse": np.sqrt(mean_squared_error(y_true, predictions)),
        "r2": r2_score(y_true, predictions),
    }


def main():
    # ---------------------------------------------------------
    # 1. Configurar MLflow local
    # ---------------------------------------------------------
    db_path = PROJ_ROOT / "mlflow.db"
    tracking_uri = f"sqlite:///{db_path.as_posix()}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    # ---------------------------------------------------------
    # 2. Cargar dataset P1
    # ---------------------------------------------------------
    frame = pd.read_parquet(FEATURES_PATH)
    frame["target_date"] = pd.to_datetime(frame["target_date"])

    # Mismo conjunto de evaluación para A y B:
    # octubre de 2026.
    test = frame[
        (frame["target_date"] >= "2026-10-01")
        & (frame["target_date"] <= "2026-10-31")
    ].copy()

    test = test.dropna(subset=[P1_TARGET])

    X_test = test[P1_FEATURES].astype("float64")
    y_test = test[P1_TARGET].astype("float64")

    print("=== A/B TEST P1 ===")
    print(f"Observaciones de prueba: {len(test):,}")
    print(
        "Periodo:",
        test["target_date"].min(),
        "a",
        test["target_date"].max(),
    )

    # ---------------------------------------------------------
    # 3. Cargar ambos modelos desde MLflow Registry
    # ---------------------------------------------------------
    model_a_uri = f"models:/{MODEL_NAME}/{MODEL_A_VERSION}"
    model_b_uri = f"models:/{MODEL_NAME}/{MODEL_B_VERSION}"

    print("\nCargando Modelo A:", model_a_uri)
    model_a = mlflow.sklearn.load_model(model_a_uri)

    print("Cargando Modelo B:", model_b_uri)
    model_b = mlflow.sklearn.load_model(model_b_uri)

    # ---------------------------------------------------------
    # 4. Predicciones
    # ---------------------------------------------------------
    predictions_a = model_a.predict(X_test)
    predictions_b = model_b.predict(X_test)

    # ---------------------------------------------------------
    # 5. Métricas
    # ---------------------------------------------------------
    metrics_a = calculate_metrics(y_test, predictions_a)
    metrics_b = calculate_metrics(y_test, predictions_b)

    results = pd.DataFrame(
        [
            {
                "model": "A_historical_window",
                "mlflow_version": MODEL_A_VERSION,
                **metrics_a,
            },
            {
                "model": "B_recent_window",
                "mlflow_version": MODEL_B_VERSION,
                **metrics_b,
            },
        ]
    )

    # ---------------------------------------------------------
    # 6. Determinar ganador por MAE
    # ---------------------------------------------------------
    if metrics_a["mae"] < metrics_b["mae"]:
        winner = "A"
    elif metrics_b["mae"] < metrics_a["mae"]:
        winner = "B"
    else:
        winner = "TIE"

    mae_difference = metrics_a["mae"] - metrics_b["mae"]

    if metrics_a["mae"] != 0:
        mae_improvement_b_pct = (
            mae_difference / metrics_a["mae"]
        ) * 100
    else:
        mae_improvement_b_pct = 0.0

    # ---------------------------------------------------------
    # 7. Guardar resultados
    # ---------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results.to_csv(RESULTS_PATH, index=False)

    predictions = pd.DataFrame(
        {
            "target_date": test["target_date"].astype(str),
            "station_id": test["station_id"].astype(str),
            "actual": y_test.to_numpy(),
            "prediction_model_a": predictions_a,
            "prediction_model_b": predictions_b,
        }
    )

    predictions.to_csv(PREDICTIONS_PATH, index=False)

    # ---------------------------------------------------------
    # 8. Mostrar resultados
    # ---------------------------------------------------------
    print("\n=== RESULTADOS ===")
    print(results.to_string(index=False))

    print("\n=== DECISION ===")
    print("Ganador por MAE:", winner)

    print(
        "Diferencia MAE (A - B):",
        f"{mae_difference:.4f}",
    )

    print(
        "Mejora relativa de B sobre A:",
        f"{mae_improvement_b_pct:.2f}%",
    )

    print("\nResultados guardados en:")
    print(RESULTS_PATH)

    print("\nPredicciones guardadas en:")
    print(PREDICTIONS_PATH)


if __name__ == "__main__":
    main()