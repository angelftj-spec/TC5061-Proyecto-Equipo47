
"""Configuración central del proyecto MiBici."""

from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# ---------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------
PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_TRIPS_DIR = RAW_DATA_DIR / "trips"
RAW_STATIONS_DIR = RAW_DATA_DIR / "stations"

INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_TRIPS_DIR = PROCESSED_DATA_DIR / "trips"
PROCESSED_STATIONS_DIR = PROCESSED_DATA_DIR / "stations"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
REFERENCES_DIR = PROJ_ROOT / "references"

# Manifiesto de metadatos y hashes. Este archivo se versionará en Git;
# los archivos de datos permanecerán excluidos del repositorio.
DATA_MANIFEST_PATH = REFERENCES_DIR / "data_manifest.json"

# ---------------------------------------------------------------------
# Fuente oficial de MiBici
# ---------------------------------------------------------------------
MIBICI_OPEN_DATA_URL = "https://www.mibici.net/es/datos-abiertos/"

# Rango inclusivo de meses que se propone descargar.
DATA_START_MONTH = "2023-01"
DATA_END_MONTH = "2026-09"

# Versión de la nomenclatura de estaciones.
STATION_NOMENCLATURE_MONTH = "2026-09"

# ---------------------------------------------------------------------
# Partición temporal: no utilizar información futura al entrenar.
# ---------------------------------------------------------------------
TRAIN_START_MONTH = DATA_START_MONTH
TRAIN_END_MONTH = "2025-12"

VALIDATION_START_MONTH = "2026-01"
VALIDATION_END_MONTH = "2026-06"

TEST_START_MONTH = "2026-07"
TEST_END_MONTH = DATA_END_MONTH

# ---------------------------------------------------------------------
# Reproducibilidad
# ---------------------------------------------------------------------
RANDOM_SEED = 42
REQUEST_TIMEOUT_SECONDS = 120

DATA_VERSION = (
    f"mibici-{DATA_START_MONTH}_to_{DATA_END_MONTH}"
    f"-stations-{STATION_NOMENCLATURE_MONTH}"
)

MLFLOW_EXPERIMENT_NAME = "mibici-equipo47"

logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

# Configuración de Loguru compatible con tqdm.
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda message: tqdm.write(str(message), end=""), colorize=True)
except ModuleNotFoundError:
    pass
