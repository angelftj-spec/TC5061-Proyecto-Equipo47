
"""Carga e inferencia del modelo de demanda MiBici P1."""

import json
import math
import os
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from mibici.config import PROJ_ROOT
from mibici.modeling.predict import P1_FEATURES, P1_MODEL_NAME
from mibici.api.schemas import DemandFeatures, DemandPrediction


class DemandModelService:
    """Servicio de inferencia para el modelo P1 registrado en MLflow."""

    def __init__(self, model, version: str, run_id: str | None = None):
        self.model = model
        self.model_name = P1_MODEL_NAME
        self.model_version = str(version)
        self.run_id = run_id

    @classmethod
    def load(cls):
        """Carga el modelo desde MLflow o desde un artefacto empaquetado."""

        # Docker: cargar el modelo exportado.
        packaged_path = os.getenv("MIBICI_MODEL_PATH")

        if packaged_path:
            path = Path(packaged_path)

            if not path.is_dir():
                raise FileNotFoundError(
                    f"No existe el directorio del modelo: {path}"
                )

            metadata_path = path / "deployment_metadata.json"

            if not metadata_path.is_file():
                raise FileNotFoundError(
                    f"No existe el archivo de metadatos: {metadata_path}"
                )

            metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )

            if metadata.get("model_name") != P1_MODEL_NAME:
                raise ValueError(
                    "El nombre del modelo en los metadatos no coincide."
                )

            version = str(metadata["model_version"])
            run_id = metadata.get("run_id")

            model = mlflow.pyfunc.load_model(str(path))

            print(f"Modelo empaquetado cargado: {path}")
            print(f"Versión: {version}")
            print(f"Run ID: {run_id}")

            return cls(model, version, run_id)

        # Desarrollo local: utilizar el Registry de MLflow.
        db_path = PROJ_ROOT / "mlflow.db"
        default_uri = f"sqlite:///{db_path.as_posix()}"

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", default_uri)
        registry_uri = os.getenv("MLFLOW_REGISTRY_URI", tracking_uri)

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(registry_uri)

        client = MlflowClient(registry_uri=registry_uri)

        registered_version = client.get_model_version_by_alias(
            P1_MODEL_NAME,
            "champion",
        )

        version = str(registered_version.version)
        run_id = registered_version.run_id
        model_uri = f"models:/{P1_MODEL_NAME}/{version}"

        model = mlflow.pyfunc.load_model(model_uri)

        print(f"Modelo cargado desde Registry: {model_uri}")

        return cls(model, version, run_id)

    def predict(self, features: DemandFeatures) -> DemandPrediction:
        """Convierte la petición a DataFrame y devuelve la predicción."""

        values = features.model_dump()

        frame = pd.DataFrame(
            [{name: values[name] for name in P1_FEATURES}],
            columns=P1_FEATURES,
        ).astype("float64")

        predictions = self.model.predict(frame)
        result = float(predictions[0])

        if not math.isfinite(result) or result < 0:
            raise RuntimeError(
                "El modelo devolvió una predicción inválida."
            )

        return DemandPrediction(
            predicted_trips_next_day=result,
            model_name=self.model_name,
            model_version=self.model_version,
        )
