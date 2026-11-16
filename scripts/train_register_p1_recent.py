
"""Reproduce y registra localmente el modelo final P1 de MiBici."""

from pathlib import Path
import hashlib

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from mibici.config import MLFLOW_EXPERIMENT_NAME, PROJ_ROOT, RANDOM_SEED
from mibici.modeling.run_final_models import P1_FEATURES, P1_TARGET


MODEL_NAME = "mibici-equipo47-p1-demanda"
FEATURES_PATH = PROJ_ROOT / "data/processed/features/p1_regression.parquet"


def main():
    # 1. Configurar Registry y Tracking locales.
    db_path = PROJ_ROOT / "mlflow.db"
    tracking_uri = f"sqlite:///{db_path.as_posix()}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    frame = pd.read_parquet(FEATURES_PATH)

    frame["target_date"] = pd.to_datetime(frame["target_date"])

    train = frame[
        (frame["target_date"] >= "2025-01-01")
        & (frame["target_date"] <= "2026-06-30")
    ].copy()

    validation = frame[
        (frame["target_date"] >= "2026-07-01")
        & (frame["target_date"] <= "2026-09-30")
    ].copy()

    test = frame[
        (frame["target_date"] >= "2026-10-01")
        & (frame["target_date"] <= "2026-10-31")
    ].copy()

    train = train.dropna(subset=[P1_TARGET])
    validation = validation.dropna(subset=[P1_TARGET])
    test = test.dropna(subset=[P1_TARGET])

    X_train = train[P1_FEATURES].astype("float64")
    y_train = train[P1_TARGET].astype("float64")

    # 3. Reproducir los hiperparámetros finales del equipo.
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("regressor", RandomForestRegressor(
            n_estimators=80,
            max_depth=14,
            min_samples_leaf=5,
            random_state=RANDOM_SEED,
            n_jobs=2,
        )),
    ])

    print("Entrenando Random Forest P1...")
    print(f"Registros de entrenamiento: {len(train):,}")

    with mlflow.start_run(run_name="P1-recent-window-candidate") as run:
        mlflow.set_tags({
            "question": "P1",
            "team": "Equipo47",
            "stage_of_work": "ab_testing",
            "model_origin": "recent_window_retrain",
            "data_source": "p1_regression.parquet",
            "ab_variant": "B",
        })

        mlflow.log_params({
            "n_estimators": 80,
            "max_depth": 14,
            "min_samples_leaf": 5,
            "random_seed": RANDOM_SEED,
            "n_jobs": 2,
            "features_used": len(P1_FEATURES),
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "target_date_min": str(frame["target_date"].min()),
            "target_date_max": str(frame["target_date"].max()),
            "train_start": "2025-01-01",
            "train_end": "2026-06-30",
            "validation_start": "2026-07-01",
            "validation_end": "2026-09-30",
            "test_start": "2026-10-01",
            "test_end": "2026-10-31",
        })

        # Hash del archivo efectivamente utilizado.
        with FEATURES_PATH.open("rb") as file:
            data_hash = hashlib.file_digest(file, "sha256").hexdigest()
        mlflow.set_tag("features_sha256", data_hash)

        model.fit(X_train, y_train)

        # 4. Evaluar sin entrenar sobre validación/prueba.
        for name, subset in [
            ("validation", validation),
            ("test", test),
        ]:
            X = subset[P1_FEATURES].astype("float64")
            y = subset[P1_TARGET].astype("float64")
            predictions = model.predict(X)

            mlflow.log_metrics({
                f"{name}_mae": mean_absolute_error(y, predictions),
                f"{name}_rmse": np.sqrt(
                    mean_squared_error(y, predictions)
                ),
                f"{name}_r2": r2_score(y, predictions),
            })

        # 5. Guardar modelo y esquema de inferencia en MLflow.
        example = X_train.head(5).fillna(
            X_train.median(numeric_only=True)
        )
        signature = infer_signature(
            example,
            model.predict(example),
        )

        logged_model = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=example,
            signature=signature,
            serialization_format="cloudpickle",
        )

        print(f"MLflow Run ID: {run.info.run_id}")

    # 6. Registrar una versión con nombre oficial del proyecto.
    registered = mlflow.register_model(
        model_uri=logged_model.model_uri,
        name=MODEL_NAME,
    )

    client = MlflowClient()
    client.set_registered_model_alias(
        MODEL_NAME,
        "candidate",
        registered.version,
    )
    client.set_model_version_tag(
        MODEL_NAME,
        registered.version,
        "origin",
        "recent_window_retrain",
    )

    print("\n=== REGISTRO COMPLETADO ===")
    print("Nombre:", MODEL_NAME)
    print("Versión:", registered.version)
    print("Alias: candidate")
    print("URI:", f"models:/{MODEL_NAME}/{registered.version}")
    print("Datos SHA256:", data_hash)


if __name__ == "__main__":
    main()
