
"""Exporta el modelo champion de MLflow para el contenedor."""

import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
from mlflow.tracking import MlflowClient

from mibici.config import PROJ_ROOT


MODEL_NAME = "mibici-equipo47-p1-demanda"
EXPORT_DIR = PROJ_ROOT / "models" / "p1_serving"


def main():
    # Conectar con el Registry local.
    db_path = PROJ_ROOT / "mlflow.db"
    uri = f"sqlite:///{db_path.as_posix()}"

    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)

    client = MlflowClient()

    # Resolver el alias champion.
    model_version = client.get_model_version_by_alias(
        MODEL_NAME, "champion"
    )

    version = str(model_version.version)
    model_uri = f"models:/{MODEL_NAME}/{version}"

    print("Modelo:", MODEL_NAME)
    print("Versión seleccionada:", version)
    print("Origen:", model_uri)

    if EXPORT_DIR.exists():
        raise FileExistsError(
            f"Ya existe {EXPORT_DIR}. "
            "Revisa el contenido antes de reemplazarlo."
        )

    # Descargar el modelo desde MLflow Registry.
    with TemporaryDirectory() as temporary_dir:
        downloaded_path = Path(
            mlflow.artifacts.download_artifacts(
                artifact_uri=model_uri,
                dst_path=temporary_dir,
                tracking_uri=uri,
                registry_uri=uri,
            )
        )

        if not (downloaded_path / "MLmodel").is_file():
            raise RuntimeError(
                "La descarga no contiene un modelo MLflow válido."
            )

        shutil.copytree(downloaded_path, EXPORT_DIR)

    # Conservar la trazabilidad del modelo exportado.
    metadata = {
        "model_name": MODEL_NAME,
        "model_version": version,
        "model_alias": "champion",
        "model_uri": model_uri,
        "run_id": model_version.run_id,
    }

    (EXPORT_DIR / "deployment_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    # Comprobar que el artefacto es autocontenido.
    model = mlflow.pyfunc.load_model(str(EXPORT_DIR))

    print("\n=== EXPORTACIÓN EXITOSA ===")
    print("Ubicación:", EXPORT_DIR)
    print("Versión:", version)
    print("Modelo local cargado:", model is not None)


if __name__ == "__main__":
    main()
