from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from mibici.modeling.run_experiments import (
    FEATURES_DIR,
    git_metadata,
    load_manifest,
    log_run_context,
    setup_tracking,
)

RANDOM_SEED = 42
CANDIDATE_K_VALUES = [3, 4, 5, 6, 8]
TEAM_NAME = "Equipo47"


def main() -> None:
    """Compara candidatos K-Means para P3 y registra resultados en MLflow."""
    setup_tracking()
    manifest, manifest_hash = load_manifest()
    git_info = git_metadata()

    path = FEATURES_DIR / "p3_clustering.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"No se encontró {path}. Ejecuta primero el feature engineering."
        )

    frame = pd.read_parquet(path)

    # Usar las mismas variables numéricas que el baseline para comparar justamente.
    excluded = {"station_id", "nombre_estacion"}
    numeric_columns = [
        c for c in frame.select_dtypes(include="number").columns
        if c not in excluded and frame[c].notna().any()
    ]
    if not numeric_columns:
        raise ValueError("No hay características numéricas para P3.")

    features = frame[numeric_columns].replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median")
    imputed = imputer.fit_transform(features)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(imputed)

    results = []
    parent_name = "P3-kmeans-candidate-search"

    with mlflow.start_run(run_name=parent_name) as parent_run:
        mlflow.set_tag("question", "P3")
        mlflow.set_tag("model_family", "KMeans")
        mlflow.set_tag("stage_of_work", "search")
        mlflow.set_tag("team", TEAM_NAME)
        log_run_context("P3", manifest, manifest_hash, git_info)
        mlflow.log_params({
            "candidate_k_values": ",".join(map(str, CANDIDATE_K_VALUES)),
            "random_state": RANDOM_SEED,
            "n_init": 20,
            "scaling": "SimpleImputer_median_then_StandardScaler",
            "feature_columns": ",".join(numeric_columns),
            "n_features": len(numeric_columns),
            "n_stations": len(frame),
            "excluded_columns": ",".join(sorted(excluded)),
        })

        for k in CANDIDATE_K_VALUES:
            if k >= len(frame):
                logger.warning(f"Se omite k={k}: supera el número de estaciones.")
                continue

            with mlflow.start_run(
                run_name=f"P3-kmeans-k-{k}",
                nested=True,
            ):
                model = KMeans(
                    n_clusters=k,
                    random_state=RANDOM_SEED,
                    n_init=20,
                )
                labels = model.fit_predict(scaled)
                silhouette = float(silhouette_score(scaled, labels))
                inertia = float(model.inertia_)
                counts = pd.Series(labels).value_counts().sort_index()

                mlflow.set_tag("question", "P3")
                mlflow.set_tag("model_family", "KMeans")
                mlflow.set_tag("stage_of_work", "candidate")
                mlflow.set_tag("team", TEAM_NAME)
                mlflow.log_params({
                    "model_type": "KMeans",
                    "n_clusters": k,
                    "n_init": 20,
                    "random_state": RANDOM_SEED,
                    "scaling": "SimpleImputer_median_then_StandardScaler",
                    "feature_columns": ",".join(numeric_columns),
                    "n_features": len(numeric_columns),
                    "n_stations": len(frame),
                })
                mlflow.log_metrics({
                    "silhouette_score": silhouette,
                    "inertia": inertia,
                    "smallest_cluster_size": int(counts.min()),
                    "largest_cluster_size": int(counts.max()),
                    "cluster_size_ratio": float(counts.max() / counts.min()),
                })
                for cluster_id, count in counts.items():
                    mlflow.log_metric(
                        f"cluster_{cluster_id}_stations", int(count)
                    )

                assignments = frame.copy()
                assignments["cluster"] = labels
                artifact_columns = [
                    c for c in [
                        "station_id", "nombre_estacion", "latitud",
                        "longitud", "capacidad", "cluster",
                    ] if c in assignments.columns
                ]

                with tempfile.TemporaryDirectory(prefix=f"mibici_p3_k{k}_") as temp:
                    temp_path = Path(temp)
                    csv_path = temp_path / f"p3_clusters_k{k}.csv"
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
                    ax.set_title(f"P3: perfiles de uso por estación (k={k})")
                    fig.colorbar(scatter, ax=ax, label="Cluster")
                    fig.tight_layout()
                    plot_path = temp_path / f"p3_clusters_k{k}.png"
                    fig.savefig(plot_path, dpi=150)
                    plt.close(fig)
                    mlflow.log_artifact(str(plot_path), artifact_path="evaluation")

                results.append({
                    "model": "KMeans",
                    "n_clusters": k,
                    "silhouette_score": silhouette,
                    "inertia": inertia,
                    "smallest_cluster_size": int(counts.min()),
                    "largest_cluster_size": int(counts.max()),
                    "cluster_size_ratio": float(counts.max() / counts.min()),
                    "parent_run_id": parent_run.info.run_id,
                })

    results_df = pd.DataFrame(results).sort_values(
        "silhouette_score", ascending=False
    )
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / "p3_candidate_results.csv"
    results_df.to_csv(output_path, index=False)

    with mlflow.start_run(run_name="P3-kmeans-candidate-summary"):
        mlflow.set_tag("question", "P3")
        mlflow.set_tag("model_family", "KMeans")
        mlflow.set_tag("stage_of_work", "search_summary")
        mlflow.set_tag("team", TEAM_NAME)
        mlflow.log_artifact(str(output_path), artifact_path="reports")

    print(results_df.to_string(index=False))
    logger.success(f"Resultados guardados en {output_path}")


if __name__ == "__main__":
    main()
