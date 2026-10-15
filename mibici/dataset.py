
"""Ingesta reproducible de los datos abiertos de MiBici."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import Request, urlopen

import pandas as pd
import typer
from loguru import logger
from tqdm import tqdm

from mibici.config import (
    DATA_END_MONTH,
    DATA_MANIFEST_PATH,
    DATA_START_MONTH,
    EXTERNAL_DATA_DIR,
    MIBICI_OPEN_DATA_URL,
    PROCESSED_STATIONS_DIR,
    PROCESSED_TRIPS_DIR,
    PROJ_ROOT,
    RAW_STATIONS_DIR,
    RAW_TRIPS_DIR,
    REQUEST_TIMEOUT_SECONDS,
    STATION_NOMENCLATURE_MONTH,
    TEST_END_MONTH,
    TEST_START_MONTH,
    TRAIN_END_MONTH,
    TRAIN_START_MONTH,
    VALIDATION_END_MONTH,
    VALIDATION_START_MONTH,
)

app = typer.Typer(add_completion=False)

# Nombres diferentes usados por el portal en distintos periodos.
TRIP_COLUMN_ALIASES = {
    "id_de_viaje": "viaje_id",
    "id_viaje": "viaje_id",
    "ano_de_nacimiento": "anio_nacimiento",
    "anio_de_nacimiento": "anio_nacimiento",
    "nacimiento": "anio_nacimiento",
    "inicio_del_viaje": "inicio_viaje",
    "inicio_de_viaje": "inicio_viaje",
    "fecha_hora_inicio": "inicio_viaje",
    "fecha_inicio_viaje": "inicio_viaje",
    "fecha_de_inicio": "fecha_inicio",
    "hora_de_inicio": "hora_inicio",
    "fin_del_viaje": "fin_viaje",
    "fin_de_viaje": "fin_viaje",
    "fecha_hora_fin": "fin_viaje",
    "fecha_fin_viaje": "fin_viaje",
    "fecha_de_fin": "fecha_fin",
    "hora_de_fin": "hora_fin",
    "id_origen": "origen_id",
    "id_estacion_origen": "origen_id",
    "id_de_estacion_origen": "origen_id",
    "estacion_origen_id": "origen_id",
    "id_destino": "destino_id",
    "id_estacion_destino": "destino_id",
    "id_de_estacion_destino": "destino_id",
    "estacion_destino_id": "destino_id",
}

STATION_COLUMN_ALIASES = {
    "id": "estacion_id",
    "id_estacion": "estacion_id",
    "id_de_estacion": "estacion_id",
    "station_id": "estacion_id",
    "name": "nombre_estacion",
    "nombre": "nombre_estacion",
    "nombre_de_estacion": "nombre_estacion",
    "latitude": "latitud",
    "longitude": "longitud",
    "capacity": "capacidad",
    "dpcapacity": "capacidad",
    "dp_capacity": "capacidad",
    "station_capacity": "capacidad",
}


class MiBiciLinkParser(HTMLParser):
    """Extrae enlaces mensuales y de nomenclatura del portal oficial."""

    def __init__(self) -> None:
        super().__init__()
        self.trip_links: dict[str, str] = {}
        self.station_links: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return

        href = dict(attrs).get("href")
        if not href:
            return

        url = urljoin(MIBICI_OPEN_DATA_URL, href)
        filename = Path(unquote(urlparse(url).path)).name

        trip_match = re.fullmatch(
            r"datos_abiertos_(\d{4})_(\d{2})\.csv",
            filename,
            flags=re.IGNORECASE,
        )
        station_match = re.fullmatch(
            r"nomenclatura_(\d{4})_(\d{2})\.csv",
            filename,
            flags=re.IGNORECASE,
        )

        if trip_match:
            year, month = trip_match.groups()
            self.trip_links[f"{year}-{month}"] = url

        if station_match:
            year, month = station_match.groups()
            self.station_links[f"{year}-{month}"] = url


def normalize_column_name(value: object) -> str:
    """Normaliza una columna a minúsculas ASCII con separadores snake_case."""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def normalize_columns(
    frame: pd.DataFrame,
    aliases: dict[str, str],
) -> pd.DataFrame:
    """Normaliza encabezados y aplica equivalencias conocidas."""
    result = frame.copy()
    names = []

    for column in result.columns:
        normalized = normalize_column_name(column)
        names.append(aliases.get(normalized, normalized))

    if len(names) != len(set(names)):
        duplicates = sorted(
            {name for name in names if names.count(name) > 1}
        )
        raise ValueError(
            f"Hay columnas duplicadas después de normalizarlas: {duplicates}"
        )

    result.columns = names
    return result


def fetch_download_index() -> MiBiciLinkParser:
    """Lee el portal oficial y construye el índice de archivos descargables."""
    request = Request(
        MIBICI_OPEN_DATA_URL,
        headers={"User-Agent": "TC5061-Proyecto-Equipo47/1.0"},
    )

    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            html = response.read().decode("utf-8", errors="replace")
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(
            f"No se pudo consultar el portal MiBici: {exc}"
        ) from exc

    parser = MiBiciLinkParser()
    parser.feed(html)

    if not parser.trip_links:
        raise RuntimeError(
            "No se encontraron enlaces CSV de viajes en el portal. "
            "La página podría haber cambiado de estructura."
        )

    return parser


def iter_months(start_month: str, end_month: str) -> list[str]:
    """Genera todos los meses de un rango inclusivo YYYY-MM."""
    try:
        start = pd.Period(start_month, freq="M")
        end = pd.Period(end_month, freq="M")
    except ValueError as exc:
        raise ValueError(
            "Las fechas deben tener formato YYYY-MM, por ejemplo 2023-01."
        ) from exc

    if start > end:
        raise ValueError("El mes inicial no puede ser posterior al final.")

    return [str(period) for period in pd.period_range(start, end, freq="M")]


def download_file(url: str, destination: Path, force: bool = False) -> None:
    """Descarga un archivo si no existe y lo guarda de manera atómica."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.is_file() and destination.stat().st_size > 0 and not force:
        logger.info(f"Se reutiliza el archivo descargado: {destination.name}")
        return

    temporary_path = Path(f"{destination}.part")
    request = Request(
        url,
        headers={"User-Agent": "TC5061-Proyecto-Equipo47/1.0"},
    )

    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            with temporary_path.open("wb") as output:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    output.write(block)

        if temporary_path.stat().st_size == 0:
            raise RuntimeError(f"El archivo descargado está vacío: {url}")

        with temporary_path.open("rb") as stream:
            preview = stream.read(512).lstrip().lower()

        if preview.startswith((b"<!doctype html", b"<html")):
            raise RuntimeError(
                f"El servidor devolvió HTML en lugar de un CSV: {url}"
            )

        temporary_path.replace(destination)

    except (HTTPError, URLError, TimeoutError, OSError, RuntimeError) as exc:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Falló la descarga de {url}: {exc}"
        ) from exc


def read_csv_robust(path: Path) -> pd.DataFrame:
    """Lee CSV con delimitador autodetectado y varias codificaciones."""
    last_error: Exception | None = None

    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            frame = pd.read_csv(
                path,
                sep=None,
                engine="python",
                encoding=encoding,
            )

            if len(frame.columns) > 1:
                return frame

            # Si el delimitador no se detectó correctamente, probar los
            # delimitadores más habituales de los archivos CSV.
            for separator in (",", ";", "\t", "|"):
                candidate = pd.read_csv(
                    path,
                    sep=separator,
                    encoding=encoding,
                )
                if len(candidate.columns) > 1:
                    return candidate

            last_error = ValueError(
                f"No se detectaron varias columnas en {path.name}."
            )

        except (UnicodeError, pd.errors.ParserError, ValueError) as exc:
            last_error = exc

    raise ValueError(
        f"No fue posible interpretar {path.name}. Último error: {last_error}"
    )


def clean_identifier(series: pd.Series) -> pd.Series:
    """Normaliza identificadores sin convertirlos en números decimales."""
    result = series.astype("string").str.strip()
    result = result.str.replace(r"\.0$", "", regex=True)
    return result.replace(
        {"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA}
    )


def build_datetime(
    frame: pd.DataFrame,
    canonical_name: str,
    date_column: str,
    time_column: str,
) -> pd.Series:
    """Construye una fecha-hora desde una columna o desde fecha y hora."""
    if canonical_name in frame.columns:
        values = frame[canonical_name].astype("string").fillna("").str.strip()
    elif date_column in frame.columns:
        values = frame[date_column].astype("string").fillna("").str.strip()
        if time_column in frame.columns:
            times = frame[time_column].astype("string").fillna("").str.strip()
            values = (values + " " + times).str.strip()
    else:
        raise ValueError(
            f"No se encontró '{canonical_name}' ni la columna "
            f"'{date_column}' para construir la fecha-hora. "
            f"Columnas encontradas: {list(frame.columns)}"
        )

    # Las fechas ISO (YYYY-MM-DD) deben interpretarse sin dayfirst.
    # Las fechas con barras del portal se interpretan como DD/MM/YYYY.
    iso_mask = values.str.match(
        r"^\s*\d{4}-\d{1,2}-\d{1,2}(?:[ T].*)?$",
        na=False,
    )

    parsed = pd.Series(
        pd.NaT,
        index=values.index,
        dtype="datetime64[ns]",
    )

    if iso_mask.any():
        parsed.loc[iso_mask] = pd.to_datetime(
            values.loc[iso_mask],
            format="ISO8601",
            errors="coerce",
        )

    other_mask = ~iso_mask & values.ne("")

    if other_mask.any():
        parsed.loc[other_mask] = pd.to_datetime(
            values.loc[other_mask],
            format="mixed",
            dayfirst=True,
            errors="coerce",
        )

    return parsed



def clean_trips(raw_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Normaliza viajes y conserva solo las variables necesarias para el ML.

    Se excluyen identificadores de usuarios y variables demográficas.
    Para P1-P3 son suficientes las fechas, estaciones de origen y destino,
    y la duración derivada del viaje.
    """
    frame = normalize_columns(raw_frame, TRIP_COLUMN_ALIASES)

    required_station_columns = {"origen_id", "destino_id"}
    missing = required_station_columns.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Faltan columnas de estaciones: {sorted(missing)}. "
            f"Columnas encontradas: {list(frame.columns)}"
        )

    start = build_datetime(
        frame, "inicio_viaje", "fecha_inicio", "hora_inicio"
    )
    end = build_datetime(
        frame, "fin_viaje", "fecha_fin", "hora_fin"
    )

    origin = clean_identifier(frame["origen_id"])
    destination = clean_identifier(frame["destino_id"])

    # Para predecir inicios se necesita una hora válida y estación de origen.
    valid = start.notna() & origin.notna()
    discarded = int((~valid).sum())

    clean = pd.DataFrame(
        {
            "inicio_viaje": start.loc[valid].reset_index(drop=True),
            "fin_viaje": end.loc[valid].reset_index(drop=True),
            "origen_id": origin.loc[valid].reset_index(drop=True),
            "destino_id": destination.loc[valid].reset_index(drop=True),
        }
    )

    duration = (
        clean["fin_viaje"] - clean["inicio_viaje"]
    ).dt.total_seconds() / 60.0

    # Una duración negativa suele indicar fechas malformadas.
    clean["duracion_minutos"] = duration.where(duration >= 0)

    statistics = {
        "source_rows": int(len(raw_frame)),
        "processed_rows": int(len(clean)),
        "discarded_rows_missing_start_or_origin": discarded,
    }
    return clean, statistics


def clean_stations(raw_frame: pd.DataFrame) -> pd.DataFrame:
    """Normaliza la nomenclatura y conserva los atributos de las estaciones."""
    frame = normalize_columns(raw_frame, STATION_COLUMN_ALIASES)

    if "estacion_id" not in frame.columns:
        raise ValueError(
            "La nomenclatura no contiene un ID reconocible de estación. "
            f"Columnas encontradas: {list(frame.columns)}"
        )

    frame["estacion_id"] = clean_identifier(frame["estacion_id"])
    frame = frame.dropna(subset=["estacion_id"]).copy()

    for column in ("latitud", "longitud", "capacidad"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    # Una estación debe aparecer una sola vez en la nomenclatura.
    frame = frame.drop_duplicates(subset=["estacion_id"], keep="last")
    return frame.reset_index(drop=True)


def sha256_file(path: Path) -> str:
    """Calcula SHA-256 leyendo el archivo por bloques."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def relative_path(path: Path) -> str:
    """Representa una ruta del proyecto de forma portable."""
    return path.resolve().relative_to(PROJ_ROOT.resolve()).as_posix()


@app.command()
def main(
    start_month: str = typer.Option(
        DATA_START_MONTH, help="Primer mes a procesar (YYYY-MM)."
    ),
    end_month: str = typer.Option(
        DATA_END_MONTH, help="Último mes a procesar, incluido (YYYY-MM)."
    ),
    station_month: str = typer.Option(
        STATION_NOMENCLATURE_MONTH,
        help="Mes de la nomenclatura de estaciones (YYYY-MM).",
    ),
    force_download: bool = typer.Option(
        False,
        "--force-download",
        help="Volver a descargar los CSV aunque ya existan localmente.",
    ),
) -> None:
    """Descarga, limpia, convierte y versiona los datos de MiBici."""
    months = iter_months(start_month, end_month)
    station_months = iter_months(station_month, station_month)

    logger.info("Consultando enlaces del portal oficial de MiBici...")
    index = fetch_download_index()

    missing_months = [
        month for month in months if month not in index.trip_links
    ]
    if missing_months:
        raise RuntimeError(
            "No hay enlaces publicados para estos meses: "
            f"{missing_months}. No se continuará para evitar una ingesta "
            "incompleta."
        )

    selected_station_month = station_months[0]
    if selected_station_month not in index.station_links:
        raise RuntimeError(
            f"No se encontró nomenclatura para {selected_station_month}. "
            f"Meses disponibles: {sorted(index.station_links)}"
        )

    trip_manifest: list[dict[str, object]] = []

    for month in tqdm(months, desc="Procesando viajes por mes"):
        year, month_number = month.split("-")
        filename = f"datos_abiertos_{year}_{month_number}.csv"
        raw_path = RAW_TRIPS_DIR / filename
        output_path = PROCESSED_TRIPS_DIR / f"viajes_{year}_{month_number}.parquet"
        source_url = index.trip_links[month]

        download_file(source_url, raw_path, force=force_download)

        raw_frame = read_csv_robust(raw_path)
        if raw_frame.empty:
            raise RuntimeError(f"El CSV no contiene registros: {raw_path}")

        cleaned, stats = clean_trips(raw_frame)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned.to_parquet(
            output_path,
            index=False,
            engine="pyarrow",
            compression="snappy",
        )

        trip_manifest.append(
            {
                "month": month,
                "source_url": source_url,
                "raw_path": relative_path(raw_path),
                "raw_sha256": sha256_file(raw_path),
                "source_rows": stats["source_rows"],
                "processed_path": relative_path(output_path),
                "processed_sha256": sha256_file(output_path),
                "processed_rows": stats["processed_rows"],
                "discarded_rows_missing_start_or_origin": (
                    stats["discarded_rows_missing_start_or_origin"]
                ),
            }
        )

        logger.info(
            f"{month}: {stats['source_rows']:,} filas de origen; "
            f"{stats['processed_rows']:,} filas procesadas."
        )

    # Descargar y procesar la nomenclatura de estaciones.
    station_url = index.station_links[selected_station_month]
    station_year, station_number = selected_station_month.split("-")
    station_filename = f"nomenclatura_{station_year}_{station_number}.csv"
    station_raw_path = RAW_STATIONS_DIR / station_filename
    station_output_path = (
        PROCESSED_STATIONS_DIR
        / f"nomenclatura_{station_year}_{station_number}.parquet"
    )

    download_file(station_url, station_raw_path, force=force_download)
    raw_stations = read_csv_robust(station_raw_path)

    if raw_stations.empty:
        raise RuntimeError("La nomenclatura de estaciones está vacía.")

    stations = clean_stations(raw_stations)
    station_output_path.parent.mkdir(parents=True, exist_ok=True)
    stations.to_parquet(
        station_output_path,
        index=False,
        engine="pyarrow",
        compression="snappy",
    )

    data_version = (
        f"mibici-{start_month}_to_{end_month}"
        f"-stations-{selected_station_month}"
    )

    manifest = {
        "manifest_schema_version": 1,
        "data_version": data_version,
        "source_portal": MIBICI_OPEN_DATA_URL,
        "trips_period": {
            "start_month": start_month,
            "end_month": end_month,
            "months_expected": len(months),
        },
        "temporal_split": {
            "train": {"start": TRAIN_START_MONTH, "end": TRAIN_END_MONTH},
            "validation": {
                "start": VALIDATION_START_MONTH,
                "end": VALIDATION_END_MONTH,
            },
            "test": {"start": TEST_START_MONTH, "end": TEST_END_MONTH},
        },
        "trip_files": trip_manifest,
        "stations": {
            "month": selected_station_month,
            "source_url": station_url,
            "raw_path": relative_path(station_raw_path),
            "raw_sha256": sha256_file(station_raw_path),
            "source_rows": int(len(raw_stations)),
            "processed_path": relative_path(station_output_path),
            "processed_sha256": sha256_file(station_output_path),
            "processed_rows": int(len(stations)),
        },
        "summary": {
            "months_processed": len(trip_manifest),
            "total_source_trip_rows": sum(
                int(item["source_rows"]) for item in trip_manifest
            ),
            "total_processed_trip_rows": sum(
                int(item["processed_rows"]) for item in trip_manifest
            ),
            "total_discarded_trip_rows": sum(
                int(item["discarded_rows_missing_start_or_origin"])
                for item in trip_manifest
            ),
            "stations_processed": int(len(stations)),
        },
    }

    DATA_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    logger.success(f"Ingesta terminada. Versión de datos: {data_version}")
    logger.success(
        f"Viajes procesados: "
        f"{manifest['summary']['total_processed_trip_rows']:,}"
    )
    logger.success(
        f"Estaciones procesadas: {manifest['summary']['stations_processed']:,}"
    )
    logger.success(f"Manifiesto: {DATA_MANIFEST_PATH}")


if __name__ == "__main__":
    app()