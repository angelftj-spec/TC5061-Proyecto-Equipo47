"""Entrenamiento y registro de los modelos finales P1 y P2."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from mlflow.models import infer_signature

from mibici.config import DATA_MANIFEST_PATH, RANDOM_SEED
from mibici.modeling.run_experiments import (
    FEATURES_DIR,
    setup_tracking,
    load_manifest,
    git_metadata,
    log_run_context,
    get_supervised_splits,
)


P1_FEATURES = [
    "departures", "arrivals", "latitud", "longitud", "capacidad",
    "day_of_week", "day_of_month", "month", "is_weekend",
    "day_of_week_sin", "day_of_week_cos", "month_sin", "month_cos",
    "departures_lag_1", "departures_lag_7",
    "departures_mean_prev_7d", "departures_mean_prev_28d",
    "arrivals_lag_1", "arrivals_lag_7",
    "arrivals_mean_prev_7d", "arrivals_mean_prev_28d",
]
P1_TARGET = "target_trips_next_day"

P2_FEATURES = [
    "departures", "arrivals", "latitud", "longitud", "capacidad",
    "day_of_week", "day_of_month", "month", "is_weekend",
    "day_of_week_sin", "day_of_week_cos", "month_sin", "month_cos",
    "departures_lag_1", "departures_lag_7",
    "departures_mean_prev_7d",
    "arrivals_lag_1", "arrivals_lag_7",
    "arrivals_mean_prev_7d",
]
P2_TARGET = "target_risk_station_next_day"


def load_splits(path: Path, features: list[str], target: str) -> dict:
    """Carga datos y conserva las particiones temporales del proyecto."""
    if not path.is_file():
        raise FileNotFoundError(
            f"No existe {path}. Ejecuta primero el pipeline de features."
        )

    frame = pd.read_parquet(path)
    missing = sorted(set(features + [target]) - set(frame.columns))
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    splits = get_supervised_splits(frame)

    for name, subset in splits.items():
        splits[name] = subset.dropna(subset=[target]).copy()
        if splits[name].empty:
            raise ValueError(f"El conjunto {name} quedó vacío.")

    return splits


def run_p1_final(manifest: dict, manifest_hash: str, git_info: dict) -> None:
    """Entrena y registra el Random Forest final para P1."""
    splits = load_splits(
        FEATURES_DIR / "p1_regression.parquet",
        P1_FEATURES,
        P1_TARGET,
    )

    X_train = splits["train"][P1_FEATURES].astype(float)
    y_train = splits["train"][P1_TARGET].astype(float)

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("regressor", RandomForestRegressor(
            n_estimators=80,
            max_depth=14,
            min_samples_leaf=5,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )),
    ])
    model.fit(X_train, y_train)

    with mlflow.start_run(run_name="P1-final-random-forest"):
        log_run_context("P1", manifest, manifest_hash, git_info)
        mlflow.set_tags({
            "stage_of_work": "final",
            "model_family": "RandomForestRegressor",
            "team": "Equipo47",
        })
        mlflow.log_params({
            "model_type": "RandomForestRegressor",
            "target": P1_TARGET,
            "features_used": len(P1_FEATURES),
            "train_rows": len(splits["train"]),
            "validation_rows": len(splits["validation"]),
            "test_rows": len(splits["test"]),
            "model_n_estimators": 80,
            "model_max_depth": 14,
            "model_min_samples_leaf": 5,
            "model_random_state": RANDOM_SEED,
            "selection_reason": "Best validation MAE among evaluated P1 candidates",
        })

        for split_name in ("validation", "test"):
            subset = splits[split_name]
            X = subset[P1_FEATURES].astype(float)
            y = subset[P1_TARGET].astype(float)
            predictions = model.predict(X)

            mlflow.log_metrics({
                f"{split_name}_mae": float(mean_absolute_error(y, predictions)),
                f"{split_name}_rmse": float(
                    np.sqrt(mean_squared_error(y, predictions))
                ),
                f"{split_name}_r2": float(r2_score(y, predictions)),
            })

        example = X_train.head(5).copy()
        example = example.fillna(X_train.median(numeric_only=True))
        signature = infer_signature(example, model.predict(example))

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            serialization_format="cloudpickle",
            signature=signature,
            input_example=example,
        )

        test = splits["test"]
        predictions = model.predict(test[P1_FEATURES].astype(float))
        sample = pd.DataFrame({
            "target_date": test["target_date"].astype(str),
            "station_id": test["station_id"].astype(str),
            "actual_trips": test[P1_TARGET].astype(float).to_numpy(),
            "predicted_trips": predictions,
        })
        if len(sample) > 5000:
            sample = sample.sample(n=5000, random_state=RANDOM_SEED)

        with tempfile.TemporaryDirectory(prefix="mibici_p1_final_") as temp:
            output = Path(temp) / "p1_final_test_predictions.csv"
            sample.to_csv(output, index=False)
            mlflow.log_artifact(str(output), artifact_path="evaluation")

        mlflow.log_artifact(str(DATA_MANIFEST_PATH), artifact_path="data")

    print("P1 final registrado correctamente en MLflow.")


def run_p2_final(manifest: dict, manifest_hash: str, git_info: dict) -> None:
    """Entrena y registra el HistGradientBoosting final para P2."""
    splits = load_splits(
        FEATURES_DIR / "p2_classification.parquet",
        P2_FEATURES,
        P2_TARGET,
    )

    X_train = splits["train"][P2_FEATURES].astype(float)
    y_train = splits["train"][P2_TARGET].astype(int)

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", HistGradientBoostingClassifier(
            max_iter=100,
            max_leaf_nodes=31,
            learning_rate=0.1,
            random_state=RANDOM_SEED,
        )),
    ])
    model.fit(X_train, y_train)

    with mlflow.start_run(run_name="P2-final-hist-gradient-boosting"):
        log_run_context("P2", manifest, manifest_hash, git_info)
        mlflow.set_tags({
            "stage_of_work": "final",
            "model_family": "HistGradientBoostingClassifier",
            "team": "Equipo47",
        })
        mlflow.log_params({
            "model_type": "HistGradientBoostingClassifier",
            "target": P2_TARGET,
            "features_used": len(P2_FEATURES),
            "train_rows": len(splits["train"]),
            "validation_rows": len(splits["validation"]),
            "test_rows": len(splits["test"]),
            "model_max_iter": 100,
            "model_max_leaf_nodes": 31,
            "model_learning_rate": 0.1,
            "model_random_state": RANDOM_SEED,
            "selection_reason": "Selected using validation metrics: higher balanced accuracy and precision than Random Forest, with nearly identical validation F1; test metrics are reserved for final evaluation.",
        })

        for split_name in ("validation", "test"):
            subset = splits[split_name]
            X = subset[P2_FEATURES].astype(float)
            y = subset[P2_TARGET].astype(int)
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)[:, 1]

            mlflow.log_metrics({
                f"{split_name}_accuracy": float(accuracy_score(y, predictions)),
                f"{split_name}_balanced_accuracy": float(
                    balanced_accuracy_score(y, predictions)
                ),
                f"{split_name}_precision": float(
                    precision_score(y, predictions, zero_division=0)
                ),
                f"{split_name}_recall": float(
                    recall_score(y, predictions, zero_division=0)
                ),
                f"{split_name}_f1": float(
                    f1_score(y, predictions, zero_division=0)
                ),
                f"{split_name}_roc_auc": float(roc_auc_score(y, probabilities)),
                f"{split_name}_average_precision": float(
                    average_precision_score(y, probabilities)
                ),
            })

        example = X_train.head(5).copy()
        example = example.fillna(X_train.median(numeric_only=True))
        signature = infer_signature(example, model.predict(example))

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            serialization_format="cloudpickle",
            signature=signature,
            input_example=example,
        )

        test = splits["test"]
        X_test = test[P2_FEATURES].astype(float)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        sample = pd.DataFrame({
            "target_date": test["target_date"].astype(str),
            "station_id": test["station_id"].astype(str),
            "actual_risk": test[P2_TARGET].astype(int).to_numpy(),
            "predicted_risk": predictions,
            "risk_probability": probabilities,
        })
        if len(sample) > 5000:
            sample = sample.sample(n=5000, random_state=RANDOM_SEED)

        with tempfile.TemporaryDirectory(prefix="mibici_p2_final_") as temp:
            output = Path(temp) / "p2_final_test_predictions.csv"
            sample.to_csv(output, index=False)
            mlflow.log_artifact(str(output), artifact_path="evaluation")

        mlflow.log_artifact(str(DATA_MANIFEST_PATH), artifact_path="data")

    print("P2 final registrado correctamente en MLflow.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrena y registra los modelos finales P1 y P2."
    )
    parser.add_argument(
        "--question",
        choices=["P1", "P2", "ALL"],
        default="ALL",
        help="Modelo final que se desea entrenar.",
    )
    args = parser.parse_args()

    manifest, manifest_hash = load_manifest()
    git_info = git_metadata()
    setup_tracking()

    if args.question in ("P1", "ALL"):
        run_p1_final(manifest, manifest_hash, git_info)

    if args.question in ("P2", "ALL"):
        run_p2_final(manifest, manifest_hash, git_info)

    print("Ejecuciones finales solicitadas terminadas.")


if __name__ == "__main__":
    main()
