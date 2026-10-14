
"""Experimentos baseline reproducibles de MiBici con MLflow."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import typer
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

from mibici.config import (
    DATA_END_MONTH,
    DATA_MANIFEST_PATH,
    DATA_START_MONTH,
    FIGURES_DIR,
    MLFLOW_EXPERIMENT_NAME,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
    RANDOM_SEED,
    TEST_END_MONTH,
    TEST_START_MONTH,
    TRAIN_END_MONTH,
    TRAIN_START_MONTH,
    VALIDATION_END_MONTH,
    VALIDATION_START_MONTH,
)

app = typer.Typer(add_completion=False)
FEATURES_DIR = PROCESSED_DATA_DIR / "features"


def sha256_file(path: Path) -> str:
    """Calcula el hash SHA-256 de un archivo por bloques."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_metadata() -> dict[str, str]:
    """Obtiene el commit, la rama y si existen cambios locales pendientes."""
    commit_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJ_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    status_result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJ_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    branch_result = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=PROJ_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    return {
        "git_commit": (
            commit_result.stdout.strip()
            if commit_result.returncode == 0
            else "unknown"
        ),
        "git_branch": (
            branch_result.stdout.strip()
            if branch_result.returncode == 0
            else "unknown"
        ),
        "git_worktree_dirty": str(bool(status_result.stdout.strip())).lower(),
    }


def load_manifest() -> tuple[dict, str]:
    """Carga el manifiesto y devuelve sus metadatos y hash."""
    if not DATA_MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"No existe {DATA_MANIFEST_PATH}. Ejecuta primero la ingesta."
        )

    manifest = json.loads(
        DATA_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    return manifest, sha256_file(DATA_MANIFEST_PATH)


def setup_tracking() -> None:
    """Configura MLflow para guardar los experimentos localmente."""
    tracking_dir = PROJ_ROOT / "mlruns"
    tracking_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def log_run_context(
    question: str,
    manifest: dict,
    manifest_hash: str,
    git_info: dict[str, str],
) -> None:
    """Registra los metadatos necesarios para identificar cada ejecución."""
    period = manifest["trips_period"]

    mlflow.log_params(
        {
            "data_version": manifest["data_version"],
            "data_manifest_sha256": manifest_hash,
            "data_start_month": period["start_month"],
            "data_end_month": period["end_month"],
            "station_nomenclature_month": manifest["stations"]["month"],
            "random_seed": RANDOM_SEED,
            "target_train_start": TRAIN_START_MONTH,
            "target_train_end": TRAIN_END_MONTH,
            "target_validation_start": VALIDATION_START_MONTH,
            "target_validation_end": VALIDATION_END_MONTH,
            "target_test_start": TEST_START_MONTH,
            "target_test_end": TEST_END_MONTH,
        }
    )

    mlflow.set_tags(
        {
            "team": "Equipo47",
            "question": question,
            "dataset_version": manifest["data_version"],
            **git_info,
        }
    )


def temporal_split(
    frame: pd.DataFrame,
    start_month: str,
    end_month: str,
) -> pd.DataFrame:
    """Filtra las muestras según el mes de la fecha objetivo."""
    if "target_date" not in frame.columns:
        raise ValueError("El conjunto debe incluir la columna target_date.")

    dates = pd.to_datetime(frame["target_date"], errors="coerce")
    start = pd.Period(start_month, freq="M").start_time
    end = pd.Period(end_month, freq="M").end_time

    return frame.loc[(dates >= start) & (dates <= end)].copy()


def get_supervised_splits(frame: pd.DataFrame):
    """Crea los conjuntos temporales sin barajar las fechas."""
    splits = {
        "train": temporal_split(frame, TRAIN_START_MONTH, TRAIN_END_MONTH),
        "validation": temporal_split(
            frame, VALIDATION_START_MONTH, VALIDATION_END_MONTH
        ),
        "test": temporal_split(frame, TEST_START_MONTH, TEST_END_MONTH),
    }

    for name, subset in splits.items():
        if subset.empty:
            raise ValueError(
                f"El conjunto {name} está vacío. Revisa las fechas y los datos."
            )

    return splits


def placeholder_x(n_rows: int) -> np.ndarray:
    """Devuelve una columna ficticia para los modelos Dummy."""
    return np.zeros((n_rows, 1), dtype=np.float64)


def run_p1(manifest: dict, manifest_hash: str, git_info: dict[str, str]) -> None:
    """Ejecuta el baseline de regresión para los viajes del día siguiente."""
    path = FEATURES_DIR / "p1_regression.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró {path}. Ejecuta features.py.")

    frame = pd.read_parquet(path)
    splits = get_supervised_splits(frame)
    target = "target_trips_next_day"

    y_train = splits["train"][target].astype(float)
    model = DummyRegressor(strategy="mean")
    model.fit(placeholder_x(len(y_train)), y_train)

    run_name = "P1-baseline-dummy-regressor"

    with mlflow.start_run(run_name=run_name):
        log_run_context("P1", manifest, manifest_hash, git_info)
        mlflow.log_params(
            {
                "model_type": "DummyRegressor",
                "strategy": "mean",
                "target": target,
                "train_rows": len(splits["train"]),
                "validation_rows": len(splits["validation"]),
                "test_rows": len(splits["test"]),
                "features_used": "none_baseline",
            }
        )

        test_sample = None

        for split_name in ("validation", "test"):
            subset = splits[split_name]
            y_true = subset[target].astype(float)
            predictions = model.predict(placeholder_x(len(subset)))

            metrics = {
                f"{split_name}_mae": float(
                    mean_absolute_error(y_true, predictions)
                ),
                f"{split_name}_rmse": float(
                    np.sqrt(np.mean((y_true.to_numpy() - predictions) ** 2))
                ),
                f"{split_name}_r2": float(r2_score(y_true, predictions)),
            }
            mlflow.log_metrics(metrics)

            if split_name == "test":
                test_sample = pd.DataFrame(
                    {
                        "target_date": subset["target_date"].astype(str),
                        "station_id": subset["station_id"].astype(str),
                        "actual_trips": y_true,
                        "predicted_trips": predictions,
                    }
                )
                if len(test_sample) > 5000:
                    test_sample = test_sample.sample(
                        n=5000, random_state=RANDOM_SEED
                    )

        with tempfile.TemporaryDirectory(prefix="mibici_p1_") as temp:
            temp_path = Path(temp)
            csv_path = temp_path / "p1_test_predictions_sample.csv"
            test_sample.to_csv(csv_path, index=False)
            mlflow.log_artifact(str(csv_path), artifact_path="evaluation")

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(
                test_sample["actual_trips"],
                test_sample["predicted_trips"],
                alpha=0.25,
                s=10,
            )
            ax.set_xlabel("Viajes reales del día siguiente")
            ax.set_ylabel("Viajes predichos")
            ax.set_title("P1: baseline de regresión")
            fig.tight_layout()

            plot_path = temp_path / "p1_actual_vs_predicted.png"
            fig.savefig(plot_path, dpi=140)
            plt.close(fig)
            mlflow.log_artifact(str(plot_path), artifact_path="evaluation")

    logger.success("Run de P1 registrado en MLflow.")


def classification_metrics(
    y_true: pd.Series,
    predictions: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    """Calcula las métricas principales de clasificación binaria."""
    return {
        f"{prefix}_accuracy": float(accuracy_score(y_true, predictions)),
        f"{prefix}_balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        f"{prefix}_precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        f"{prefix}_recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        f"{prefix}_f1": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
    }


def run_p2(manifest: dict, manifest_hash: str, git_info: dict[str, str]) -> None:
    """Ejecuta el baseline para la etiqueta aproximada de riesgo."""
    path = FEATURES_DIR / "p2_classification.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró {path}. Ejecuta features.py.")

    frame = pd.read_parquet(path)
    splits = get_supervised_splits(frame)
    target = "target_risk_station_next_day"

    y_train = splits["train"][target].astype(int)
    model = DummyClassifier(strategy="most_frequent")
    model.fit(placeholder_x(len(y_train)), y_train)

    run_name = "P2-baseline-dummy-classifier"

    with mlflow.start_run(run_name=run_name):
        log_run_context("P2", manifest, manifest_hash, git_info)
        mlflow.log_params(
            {
                "model_type": "DummyClassifier",
                "strategy": "most_frequent",
                "target": target,
                "risk_capacity_fraction": 0.20,
                "train_rows": len(splits["train"]),
                "validation_rows": len(splits["validation"]),
                "test_rows": len(splits["test"]),
                "features_used": "none_baseline",
            }
        )

        test_predictions = None
        test_y = None
        test_subset = None

        for split_name in ("validation", "test"):
            subset = splits[split_name]
            y_true = subset[target].astype(int)
            predictions = model.predict(placeholder_x(len(subset)))

            mlflow.log_metrics(
                classification_metrics(y_true, predictions, split_name)
            )

            if split_name == "test":
                test_predictions = predictions
                test_y = y_true
                test_subset = subset

        with tempfile.TemporaryDirectory(prefix="mibici_p2_") as temp:
            temp_path = Path(temp)

            csv_path = temp_path / "p2_test_predictions_sample.csv"
            sample = pd.DataFrame(
                {
                    "target_date": test_subset["target_date"].astype(str),
                    "station_id": test_subset["station_id"].astype(str),
                    "actual_risk": test_y,
                    "predicted_risk": test_predictions,
                }
            )
            if len(sample) > 5000:
                sample = sample.sample(n=5000, random_state=RANDOM_SEED)
            sample.to_csv(csv_path, index=False)
            mlflow.log_artifact(str(csv_path), artifact_path="evaluation")

            matrix = confusion_matrix(test_y, test_predictions, labels=[0, 1])
            fig, ax = plt.subplots(figsize=(6, 5))
            from sklearn.metrics import ConfusionMatrixDisplay

            ConfusionMatrixDisplay(
                confusion_matrix=matrix,
                display_labels=["Sin riesgo proxy", "Riesgo proxy"],
            ).plot(ax=ax, values_format="d", colorbar=False)
            ax.set_title("P2: matriz de confusión en prueba")
            fig.tight_layout()

            plot_path = temp_path / "p2_confusion_matrix.png"
            fig.savefig(plot_path, dpi=140)
            plt.close(fig)
            mlflow.log_artifact(str(plot_path), artifact_path="evaluation")

    logger.success("Run de P2 registrado en MLflow.")


def run_p3(
    manifest: dict,
    manifest_hash: str,
    git_info: dict[str, str],
    n_clusters: int,
) -> None:
    """Ejecuta K-Means y registra métricas y un mapa de las estaciones."""
    path = FEATURES_DIR / "p3_clustering.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró {path}. Ejecuta features.py.")

    frame = pd.read_parquet(path)

    if n_clusters < 2 or n_clusters >= len(frame):
        raise ValueError("n_clusters debe ser al menos 2 y menor que estaciones.")

    numeric_columns = [
        column
        for column in frame.select_dtypes(include="number").columns
        if frame[column].notna().any()
    ]
    if not numeric_columns:
        raise ValueError("No se encontraron variables numéricas para clustering.")

    features = frame[numeric_columns].replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median")
    imputed = imputer.fit_transform(features)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(imputed)

    model = KMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_SEED,
        n_init=20,
    )
    labels = model.fit_predict(scaled)

    assignments = frame.copy()
    assignments["cluster"] = labels

    # La silueta necesita al menos dos clusters y menos clusters que muestras.
    from sklearn.metrics import silhouette_score

    silhouette = float(silhouette_score(scaled, labels))
    run_name = "P3-baseline-kmeans"

    with mlflow.start_run(run_name=run_name):
        log_run_context("P3", manifest, manifest_hash, git_info)
        mlflow.log_params(
            {
                "model_type": "KMeans",
                "n_clusters": n_clusters,
                "n_init": 20,
                "random_state": RANDOM_SEED,
                "scaling": "SimpleImputer_median_then_StandardScaler",
                "feature_columns": ",".join(numeric_columns),
                "n_features": len(numeric_columns),
                "n_stations": len(assignments),
            }
        )
        mlflow.log_metrics(
            {
                "silhouette_score": silhouette,
                "inertia": float(model.inertia_),
            }
        )

        counts = pd.Series(labels).value_counts().sort_index()
        for cluster_id, count in counts.items():
            mlflow.log_metric(f"cluster_{cluster_id}_stations", int(count))

        artifact_columns = [
            column
            for column in (
                "station_id",
                "nombre_estacion",
                "latitud",
                "longitud",
                "capacidad",
                "cluster",
            )
            if column in assignments.columns
        ]

        with tempfile.TemporaryDirectory(prefix="mibici_p3_") as temp:
            temp_path = Path(temp)

            csv_path = temp_path / "p3_station_clusters.csv"
            assignments[artifact_columns].to_csv(csv_path, index=False)
            mlflow.log_artifact(str(csv_path), artifact_path="evaluation")

            fig, ax = plt.subplots(figsize=(9, 7))
            scatter = ax.scatter(
                assignments["longitud"],
                assignments["latitud"],
                c=labels,
                cmap="tab10",
                s=18,
                alpha=0.85,
            )
            ax.set_xlabel("Longitud")
            ax.set_ylabel("Latitud")
            ax.set_title(f"P3: estaciones agrupadas con K-Means (k={n_clusters})")
            fig.colorbar(scatter, ax=ax, label="Cluster")
            fig.tight_layout()

            plot_path = temp_path / "p3_station_clusters.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            mlflow.log_artifact(str(plot_path), artifact_path="evaluation")

    logger.success("Run de P3 registrado en MLflow.")


@app.command()
def main(
    question: str = typer.Option(
        "all",
        "--question",
        help="Experimento a ejecutar: all, P1, P2 o P3.",
    ),
    n_clusters: int = typer.Option(
        4,
        "--n-clusters",
        min=2,
        max=20,
        help="Número de clusters para P3.",
    ),
) -> None:
    """Ejecuta los baselines y los registra en MLflow Tracking."""
    question = question.upper()
    if question not in {"ALL", "P1", "P2", "P3"}:
        raise typer.BadParameter("Usa all, P1, P2 o P3.")

    manifest, manifest_hash = load_manifest()
    git_info = git_metadata()

    setup_tracking()
    logger.info(f"Experimento MLflow: {MLFLOW_EXPERIMENT_NAME}")
    logger.info(f"Versión de datos: {manifest['data_version']}")
    logger.info(f"Commit registrado: {git_info['git_commit']}")
    logger.info(f"Árbol de trabajo modificado: {git_info['git_worktree_dirty']}")

    if question in {"ALL", "P1"}:
        run_p1(manifest, manifest_hash, git_info)

    if question in {"ALL", "P2"}:
        run_p2(manifest, manifest_hash, git_info)

    if question in {"ALL", "P3"}:
        run_p3(manifest, manifest_hash, git_info, n_clusters)

    logger.success("Experimentos solicitados terminados.")


if __name__ == "__main__":
    app()
