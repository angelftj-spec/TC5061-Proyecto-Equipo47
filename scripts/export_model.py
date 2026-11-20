"""Exporta el modelo champion de MLflow para el contenedor."""

import json
import shutil
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from mibici.config import PROJ_ROOT


MODEL_NAME = "mibici-equipo47-p1-demanda"
EXPORT_DIR = PROJ_ROOT / "models" / "p1_serving"
MLRUNS_DIR = PROJ_ROOT / "mlruns"


def find_model_artifacts(run_id: str) -> Path:
    """Encuentra los artefactos MLflow asociados al run_id."""

    matches = []

    for mlmodel_file in MLRUNS_DIR.rglob("MLmodel"):
        if "models" not in mlmodel_file.parts:
            continue

        content = mlmodel_file.read_text(encoding="utf-8")

        if run_id in content:
            artifact_dir = mlmodel_file.parent

            if (artifact_dir / "model.pkl").is_file():
                matches.append(artifact_dir)

    if not matches:
        raise FileNotFoundError(
            f"No se encontraron artefactos del modelo para run_id={run_id}"
        )

    if len(matches) > 1:
        raise RuntimeError(
            "Se encontraron múltiples modelos para el run_id:\n"
            + "\n".join(str(path) for path in matches)
        )

    return matches[0]


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
    run_id = model_version.run_id
    model_uri = f"models:/{MODEL_NAME}/{version}"

    print("Modelo:", MODEL_NAME)
    print("Versión seleccionada:", version)
    print("Run ID:", run_id)
    print("Origen:", model_uri)

    if EXPORT_DIR.exists():
        raise FileExistsError(
            f"Ya existe {EXPORT_DIR}. "
            "Revisa el contenido antes de reemplazarlo."
        )

    # Encontrar los artefactos del modelo en MLflow 3.
    source_dir = find_model_artifacts(run_id)

    print("Artefactos encontrados:", source_dir)

    if not (source_dir / "MLmodel").is_file():
        raise RuntimeError(
            "Los artefactos encontrados no contienen un MLmodel válido."
        )

    # Exportar modelo.
    shutil.copytree(source_dir, EXPORT_DIR)

    # Conservar trazabilidad.
    metadata = {
        "model_name": MODEL_NAME,
        "model_version": version,
        "model_alias": "champion",
        "model_uri": model_uri,
        "run_id": run_id,
    }

    (EXPORT_DIR / "deployment_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    # Comprobar que el artefacto es autocontenido.
    print("\nValidando modelo exportado...")
    model = mlflow.pyfunc.load_model(str(EXPORT_DIR))

    # Validar inferencia con el ejemplo incluido.
    example_path = EXPORT_DIR / "serving_input_example.json"

    if example_path.is_file():
        example = json.loads(
            example_path.read_text(encoding="utf-8")
        )

        split_data = example["dataframe_split"]

        import pandas as pd

        df = pd.DataFrame(
            split_data["data"],
            columns=split_data["columns"],
        )

        predictions = model.predict(df)

        print("Input de prueba:", df.shape)
        print("Predicciones generadas:", len(predictions))

    print("\n=== EXPORTACIÓN EXITOSA ===")
    print("Ubicación:", EXPORT_DIR)
    print("Versión:", version)
    print("Modelo local cargado:", model is not None)


if __name__ == "__main__":
    main()