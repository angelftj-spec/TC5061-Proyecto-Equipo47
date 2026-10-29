
# mibici/validation/data_validation.py

from __future__ import annotations

import pandas as pd
import numpy as np
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import train_test_validation, data_integrity
from requests.packages import target


def validate_integrity(
    df: pd.DataFrame,
    required_columns: list[str],
    unique_columns: list[str] | None = None,
) -> dict:
    """Comprueba estructura, valores faltantes y duplicados."""

    if df.empty:
        raise ValueError("El dataset está vacío.")

    missing_columns = sorted(set(required_columns) - set(df.columns))
    if missing_columns:
        raise ValueError(
            f"Faltan columnas obligatorias: {missing_columns}"
        )

    if df.columns.duplicated().any():
        raise ValueError("El dataset contiene nombres de columnas duplicados.")

    missing_values = df.isna().sum()
    duplicate_rows = int(df.duplicated().sum())

    duplicate_keys = 0
    if unique_columns:
        absent = sorted(set(unique_columns) - set(df.columns))
        if absent:
            raise ValueError(f"Faltan columnas de clave: {absent}")

        duplicate_keys = int(
            df.duplicated(subset=unique_columns).sum()
        )

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "dtypes": {
            col: str(dtype) for col, dtype in df.dtypes.items()
        },
        "missing_by_column": {
            col: int(count)
            for col, count in missing_values.items()
        },
        "missing_total": int(missing_values.sum()),
        "duplicate_rows": duplicate_rows,
        "duplicate_keys": duplicate_keys,
    }


def validate_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str | None = None,
):
    """
    Ejecuta la suite de DeepChecks para comparar entrenamiento y prueba.

    target debe ser el nombre de la variable objetivo para tareas
    supervisadas. Para clustering, utilizar None.
    """

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Los conjuntos de entrenamiento y prueba deben contener datos."
        )

    if train_df.columns.duplicated().any() or test_df.columns.duplicated().any():
        raise ValueError("Hay nombres de columnas duplicados.")

    if set(train_df.columns) != set(test_df.columns):
        raise ValueError(
            "Entrenamiento y prueba deben tener las mismas columnas."
        )

    # Alinear el orden de las columnas.
    test_df = test_df.loc[:, train_df.columns].copy()
    train_df = train_df.copy()

    if target is not None and target not in train_df.columns:
        raise ValueError(f"No existe la variable objetivo: {target}")

    categorical_features = [
        col
        for col in ["station_id", "nombre_estacion"]
        if col in train_df.columns and col != target
    ]

    train_dataset = Dataset(
        train_df,
        label=target,
        cat_features=categorical_features,
    )

    test_dataset = Dataset(
        test_df,
        label=target,
        cat_features=categorical_features,
    )

    suite = train_test_validation()
    result = suite.run(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
    )

    return result

def validate_clustering_data(
    frame: pd.DataFrame,
    feature_columns: list[str],
) :
    """
    Valida la integridad de los datos utilizados para clustering.

    Args:
        frame: Dataset completo de estaciones.
        feature_columns: Variables utilizadas por el algoritmo.

    Returns:
        Resultado de la suite de integridad de DeepChecks.
    """
    if frame.empty:
        raise ValueError("El dataset de clustering está vacío.")

    missing_columns = set(feature_columns) - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Faltan columnas requeridas: {sorted(missing_columns)}"
        )

    if not feature_columns:
        raise ValueError("Debe especificarse al menos una variable.")

    features = frame[feature_columns].copy()

    non_numeric = features.select_dtypes(
        exclude="number"
    ).columns.tolist()

    if non_numeric:
        raise ValueError(
            f"K-Means requiere variables numéricas. "
            f"Columnas no numéricas: {non_numeric}"
        )

    # Identificar infinitos antes de ejecutar DeepChecks.
    infinite_counts = np.isinf(features.to_numpy(dtype=float)).sum(axis=0)

    if infinite_counts.any():
        infinite_columns = [
            column
            for column, count in zip(feature_columns, infinite_counts)
            if count > 0
        ]
        raise ValueError(
            f"Se encontraron valores infinitos en: {infinite_columns}"
        )

    categorical_features = [
        col
        for col in ["nombre_estacion"]
        if col in features.columns
    ]

    dataset = Dataset(
        features,
        cat_features=categorical_features,
    )

    suite = data_integrity()
    result = suite.run(dataset)

    return result
