"""Candidatos y búsqueda de hiperparámetros para P2."""

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
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    PrecisionRecallDisplay,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mibici.config import (
    DATA_MANIFEST_PATH,
    MLFLOW_EXPERIMENT_NAME,
    RANDOM_SEED,
)
from mibici.modeling.run_experiments import (
    FEATURES_DIR,
    get_supervised_splits,
    git_metadata,
    load_manifest,
    log_run_context,
    setup_tracking,
)

TARGET = "target_risk_station_next_day"

FEATURES = [
    "departures", "arrivals", "latitud", "longitud", "capacidad",
    "day_of_week", "day_of_month", "month", "is_weekend",
    "day_of_week_sin", "day_of_week_cos", "month_sin", "month_cos",
    "departures_lag_1", "departures_lag_7",
    "departures_mean_prev_7d",
    "arrivals_lag_1", "arrivals_lag_7",
    "arrivals_mean_prev_7d",
]


def make_model(name: str, params: dict):
    if name == "logistic_regression":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                max_iter=500,
                random_state=RANDOM_SEED,
                **params,
            ),
        )

    if name == "random_forest":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                random_state=RANDOM_SEED,
                n_jobs=-1,
                **params,
            ),
        )

    if name == "hist_gradient_boosting":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                random_state=RANDOM_SEED,
                **params,
            ),
        )

    raise ValueError(f"Modelo no reconocido: {name}")


def evaluate(model, X, y, prefix: str) -> dict[str, float]:
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    return {
        f"{prefix}_accuracy": float(accuracy_score(y, predictions)),
        f"{prefix}_balanced_accuracy": float(
            balanced_accuracy_score(y, predictions)
        ),
        f"{prefix}_precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        f"{prefix}_recall": float(
            recall_score(y, predictions, zero_division=0)
        ),
        f"{prefix}_f1": float(
            f1_score(y, predictions, zero_division=0)
        ),
        f"{prefix}_roc_auc": float(roc_auc_score(y, probabilities)),
        f"{prefix}_average_precision": float(
            average_precision_score(y, probabilities)
        ),
    }


def log_candidate(
    name: str,
    params: dict,
    splits: dict,
    manifest: dict,
    manifest_hash: str,
    git_info: dict[str, str],
    nested: bool = False,
) -> dict[str, float]:
    train = splits["train"]
    validation = splits["validation"]
    test = splits["test"]

    X_train = train[FEATURES].astype(float)
    y_train = train[TARGET].astype(int)
    X_val = validation[FEATURES].astype(float)
    y_val = validation[TARGET].astype(int)
    X_test = test[FEATURES].astype(float)
    y_test = test[TARGET].astype(int)

    model = make_model(name, params)
    model.fit(X_train, y_train)

    suffix = "_".join(f"{k}-{v}" for k, v in params.items())
    run_name = f"P2-{name}-{suffix}"[:250]

    with mlflow.start_run(run_name=run_name, nested=nested):
        log_run_context("P2", manifest, manifest_hash, git_info)
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

        example = X_train.head(5).copy()
        signature = infer_signature(example, model.predict(example))

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            serialization_format="cloudpickle",
            signature=signature,
            input_example=example,
        )

        mlflow.log_artifact(str(DATA_MANIFEST_PATH), artifact_path="data")

        training_data = X_train.copy()
        training_data[TARGET] = y_train.to_numpy()
        dataset = mlflow.data.from_pandas(
            training_data,
            name="mibici-p2-training",
        )
        mlflow.log_input(dataset, context="training")

        with tempfile.TemporaryDirectory(prefix="mibici_p2_") as temp:
            temp_path = Path(temp)

            config_path = temp_path / "model_config.json"
            config_path.write_text(
                json.dumps({
                    "question": "P2",
                    "model": name,
                    "parameters": params,
                    "features": FEATURES,
                    "target": TARGET,
                    "random_seed": RANDOM_SEED,
                    "data_version": manifest["data_version"],
                }, indent=2),
                encoding="utf-8",
            )
            mlflow.log_artifact(str(config_path), artifact_path="config")

            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)[:, 1]

            matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
            fig, ax = plt.subplots(figsize=(6, 5))
            ConfusionMatrixDisplay(
                confusion_matrix=matrix,
                display_labels=["Sin riesgo proxy", "Riesgo proxy"],
            ).plot(ax=ax, values_format="d", colorbar=False)
            ax.set_title(f"P2: matriz de confusión — {name}")
            fig.tight_layout()
            plot_path = temp_path / "confusion_matrix.png"
            fig.savefig(plot_path, dpi=140)
            plt.close(fig)
            mlflow.log_artifact(str(plot_path), artifact_path="plots")

            fig, ax = plt.subplots(figsize=(7, 5))
            PrecisionRecallDisplay.from_predictions(
                y_test, probabilities, ax=ax
            )
            ax.set_title(f"P2: curva precision-recall — {name}")
            fig.tight_layout()
            plot_path = temp_path / "precision_recall.png"
            fig.savefig(plot_path, dpi=140)
            plt.close(fig)
            mlflow.log_artifact(str(plot_path), artifact_path="plots")

            sample = pd.DataFrame({
                "actual_risk": y_test.to_numpy(),
                "predicted_risk": predictions,
                "predicted_probability": probabilities,
            })
            if len(sample) > 5000:
                sample = sample.sample(5000, random_state=RANDOM_SEED)
            sample_path = temp_path / "test_predictions_sample.csv"
            sample.to_csv(sample_path, index=False)
            mlflow.log_artifact(
                str(sample_path), artifact_path="evaluation"
            )

        return val_metrics


def main() -> None:
    setup_tracking()
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    manifest, manifest_hash = load_manifest()
    git_info = git_metadata()

    path = FEATURES_DIR / "p2_classification.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"No existe {path}. Ejecuta la preparación de features."
        )

    frame = pd.read_parquet(path)
    missing = sorted(set(FEATURES + [TARGET]) - set(frame.columns))
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    splits = get_supervised_splits(frame)
    for key, subset in splits.items():
        splits[key] = subset.dropna(subset=[TARGET]).copy()

    candidates = [
        ("logistic_regression", {"C": 1.0, "class_weight": "balanced"}),
        ("random_forest", {
            "n_estimators": 80,
            "max_depth": 14,
            "min_samples_leaf": 5,
            "class_weight": "balanced",
        }),
        ("hist_gradient_boosting", {
            "max_iter": 100,
            "max_leaf_nodes": 31,
            "learning_rate": 0.1,
        }),
    ]

    results = []

    for name, params in candidates:
        metrics = log_candidate(
            name, params, splits, manifest, manifest_hash, git_info
        )
        results.append({
            "model": name,
            **{k: v for k, v in params.items()},
            **metrics,
        })

    # Búsqueda anidada: cuatro combinaciones de HistGradientBoosting.
    with mlflow.start_run(run_name="P2-hist-gradient-boosting-search"):
        mlflow.set_tags({
            "question": "P2",
            "model_family": "HistGradientBoostingClassifier",
            "stage_of_work": "search",
            "team": "Equipo47",
            **git_info,
        })
        mlflow.log_params({
            "search_strategy": "manual_grid_search",
            "selection_metric": "validation_f1",
        })

        for learning_rate in (0.05, 0.1):
            for max_leaf_nodes in (15, 31):
                params = {
                    "max_iter": 100,
                    "learning_rate": learning_rate,
                    "max_leaf_nodes": max_leaf_nodes,
                }
                metrics = log_candidate(
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
                    **metrics,
                })

    results_path = Path("reports") / "p2_candidate_results.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_frame = pd.DataFrame(results).sort_values(
        "validation_f1", ascending=False
    )
    results_frame.to_csv(results_path, index=False)

    print(results_frame.to_string(index=False))
    print(f"\nResultados guardados en {results_path}")


if __name__ == "__main__":
    main()
