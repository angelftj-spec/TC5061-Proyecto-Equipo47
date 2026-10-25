"""Modelos candidatos para P1: predicción de viajes del día siguiente."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mibici.config import DATA_MANIFEST_PATH, MLFLOW_EXPERIMENT_NAME, RANDOM_SEED
from mibici.modeling.run_experiments import (
    FEATURES_DIR,
    get_supervised_splits,
    git_metadata,
    load_manifest,
    log_run_context,
    setup_tracking,
)

TARGET = "target_trips_next_day"

FEATURES = [
    "departures", "arrivals", "latitud", "longitud", "capacidad",
    "day_of_week", "day_of_month", "month", "is_weekend",
    "day_of_week_sin", "day_of_week_cos", "month_sin", "month_cos",
    "departures_lag_1", "departures_lag_7",
    "departures_mean_prev_7d", "departures_mean_prev_28d",
    "arrivals_lag_1", "arrivals_lag_7",
    "arrivals_mean_prev_7d", "arrivals_mean_prev_28d",
]


def make_model(name: str, params: dict):
    if name == "ridge":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(**params),
        )
    if name == "random_forest":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                random_state=RANDOM_SEED,
                n_jobs=-1,
                **params,
            ),
        )
    if name == "hist_gradient_boosting":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                random_state=RANDOM_SEED,
                **params,
            ),
        )
    raise ValueError(f"Modelo no reconocido: {name}")


def evaluate(model, X, y, prefix: str) -> dict[str, float]:
    predictions = model.predict(X)
    return {
        f"{prefix}_mae": float(mean_absolute_error(y, predictions)),
        f"{prefix}_rmse": float(np.sqrt(mean_squared_error(y, predictions))),
        f"{prefix}_r2": float(r2_score(y, predictions)),
    }


def log_candidate(
    name: str,
    params: dict,
    splits: dict,
    manifest: dict,
    manifest_hash: str,
    git_info: dict[str, str],
    nested: bool = False,
) -> float:
    train = splits["train"]
    validation = splits["validation"]
    test = splits["test"]

    X_train, y_train = train[FEATURES].astype("float64"), train[TARGET].astype("float64")
    X_val, y_val = validation[FEATURES].astype("float64"), validation[TARGET].astype("float64")
    X_test, y_test = test[FEATURES].astype("float64"), test[TARGET].astype("float64")

    model = make_model(name, params)
    model.fit(X_train, y_train)

    with mlflow.start_run(run_name=f"P1-{name}", nested=nested):
        log_run_context("P1", manifest, manifest_hash, git_info)
        mlflow.set_tags({
            "model_family": name,
            "stage_of_work": "candidate",
            "team": "Equipo47",
        })
        mlflow.log_params({
            "model_type": name,
            "target": TARGET,
            "features_used": len(FEATURES),
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
            **{f"model_{k}": str(v) for k, v in params.items()},
        })

        val_metrics = evaluate(model, X_val, y_val, "validation")
        test_metrics = evaluate(model, X_test, y_test, "test")
        mlflow.log_metrics({**val_metrics, **test_metrics})

        # Firma e input de ejemplo del modelo.
        example = X_train.head(5).copy()
        example = example.astype("float64")

        medians = X_train.median(numeric_only=True)
        example = example.fillna(medians).astype("float64")

        signature = infer_signature(
            example, 
            model.predict(example),
        )

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            serialization_format="skops",
            skops_trusted_types=[
                "numpy.dtype", 
                "sklearn.tree._tree.Tree",
                "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
            ],
            signature=signature,
            input_example=example,
        )

        # Manifiesto de datos y configuración reproducible.
        mlflow.log_artifact(str(DATA_MANIFEST_PATH), artifact_path="data")
        with tempfile.TemporaryDirectory(prefix="mibici_p1_") as temp:
            temp_path = Path(temp)

            config_path = temp_path / "model_config.json"
            config_path.write_text(json.dumps({
                "question": "P1",
                "model": name,
                "parameters": params,
                "features": FEATURES,
                "target": TARGET,
                "random_seed": RANDOM_SEED,
                "data_version": manifest["data_version"],
            }, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(config_path), artifact_path="config")

            # Gráficas de predicción y residuos sobre una muestra de prueba.
            predictions = model.predict(X_test)
            sample = pd.DataFrame({
                "actual": y_test.to_numpy(),
                "predicted": predictions,
            })
            if len(sample) > 5000:
                sample = sample.sample(5000, random_state=RANDOM_SEED)

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(sample["actual"], sample["predicted"], alpha=0.25, s=8)
            ax.set_xlabel("Viajes reales")
            ax.set_ylabel("Viajes predichos")
            ax.set_title(f"P1: {name} — reales vs. predichos")
            fig.tight_layout()
            plot_path = temp_path / "actual_vs_predicted.png"
            fig.savefig(plot_path, dpi=140)
            plt.close(fig)
            mlflow.log_artifact(str(plot_path), artifact_path="plots")

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(
                sample["predicted"],
                sample["actual"] - sample["predicted"],
                alpha=0.25,
                s=8,
            )
            ax.axhline(0, linestyle="--")
            ax.set_xlabel("Viajes predichos")
            ax.set_ylabel("Residuo (real - predicho)")
            ax.set_title(f"P1: residuos — {name}")
            fig.tight_layout()
            residual_path = temp_path / "residuals.png"
            fig.savefig(residual_path, dpi=140)
            plt.close(fig)
            mlflow.log_artifact(str(residual_path), artifact_path="plots")

        # Registra los datos de entrenamiento como dataset de MLflow.
        training_data = X_train.copy()
        training_data[TARGET] = y_train.to_numpy()
        dataset = mlflow.data.from_pandas(
            training_data,
            name="mibici-p1-training",
        )
        mlflow.log_input(dataset, context="training")

        return val_metrics["validation_mae"]


def main() -> None:
    setup_tracking()
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    manifest, manifest_hash = load_manifest()
    git_info = git_metadata()

    path = FEATURES_DIR / "p1_regression.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"No existe {path}. Ejecuta la preparación de features."
        )

    frame = pd.read_parquet(path)
    missing = sorted(set(FEATURES + [TARGET]) - set(frame.columns))
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    # Se conserva exactamente el split temporal del baseline.
    splits = get_supervised_splits(frame)
    for key, subset in splits.items():
        splits[key] = subset.dropna(subset=[TARGET]).copy()

    results = []

    # Tres familias candidatas.
    candidates = [
        ("ridge", {"alpha": 1.0}),
        ("random_forest", {
            "n_estimators": 80,
            "max_depth": 14,
            "min_samples_leaf": 5,
        }),
        ("hist_gradient_boosting", {
            "max_iter": 100,
            "max_leaf_nodes": 31,
            "learning_rate": 0.1,
        }),
    ]

    for name, params in candidates:
        mae = log_candidate(
            name, params, splits, manifest, manifest_hash, git_info
        )
        results.append({"model": name, "validation_mae": mae})

    # Búsqueda de hiperparámetros: parent run y una ejecución hija por combinación.
    with mlflow.start_run(run_name="P1-hist-gradient-boosting-search"):
        mlflow.set_tags({
            "question": "P1",
            "model_family": "HistGradientBoostingRegressor",
            "stage_of_work": "search",
            "team": "Equipo47",
            **git_info,
        })
        mlflow.log_param("search_strategy", "manual_grid_search")
        mlflow.log_param("selection_metric", "validation_mae")

        for learning_rate in (0.05, 0.1):
            for max_leaf_nodes in (15, 31):
                params = {
                    "max_iter": 100,
                    "learning_rate": learning_rate,
                    "max_leaf_nodes": max_leaf_nodes,
                }
                mae = log_candidate(
                    "hist_gradient_boosting",
                    params,
                    splits,
                    manifest,
                    manifest_hash,
                    git_info,
                    nested=True,
                )
                results.append({
                    "model": "hist_gradient_boosting_search",
                    **params,
                    "validation_mae": mae,
                })

    results_path = Path("reports") / "p1_candidate_results.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).sort_values("validation_mae").to_csv(
        results_path, index=False
    )
    print(pd.DataFrame(results).sort_values("validation_mae").to_string(index=False))
    print(f"\nResultados guardados en {results_path}")


if __name__ == "__main__":
    main()
