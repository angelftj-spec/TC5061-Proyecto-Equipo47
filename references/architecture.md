# Arquitectura del proyecto MiBici — Equipo 47

El flujo comienza con los datos abiertos de MiBici, continúa con la ingesta y transformación de los viajes, genera los conjuntos de datos para las tres preguntas de Machine Learning y registra los experimentos en MLflow.

```mermaid
flowchart TD
    A["Portal oficial de datos abiertos MiBici"] --> B["mibici/dataset.py<br/>Descarga y limpieza"]

    B --> C["data/raw/<br/>CSV mensuales y nomenclatura"]
    C --> D["Validación y normalización<br/>Fechas, columnas e identificadores"]

    D --> E["data/processed/<br/>Parquet de viajes y estaciones"]
    D --> F["references/data_manifest.json<br/>Conteos y hashes SHA-256"]

    E --> G["mibici/features.py<br/>Ingeniería de variables"]

    G --> H["daily_station_flows.parquet<br/>Flujos diarios por estación"]
    G --> I["p1_regression.parquet<br/>Predicción de demanda"]
    G --> J["p2_classification.parquet<br/>Riesgo aproximado"]
    G --> K["p3_clustering.parquet<br/>Perfiles de estaciones"]

    I --> L["P1: DummyRegressor"]
    J --> M["P2: DummyClassifier"]
    K --> N["P3: K-Means"]

    L --> O["mibici/modeling/run_experiments.py"]
    M --> O
    N --> O

    O --> P["MLflow Tracking"]
    P --> Q["Base local SQLite<br/>mlflow.db"]
    P --> R["Métricas, parámetros<br/>y artefactos"]

    S["config.py<br/>Parámetros y particiones temporales"] -.-> B
    S -.-> G
    S -.-> O

    T["Git y GitHub<br/>Código, pruebas y documentación"] --- U["pytest<br/>Pruebas automatizadas"]
```

## Componentes principales

| Componente | Responsabilidad |
|---|---|
| `dataset.py` | Descarga los archivos de origen, normaliza y limpia los datos. |
| `features.py` | Construye flujos diarios y variables para P1, P2 y P3. |
| `config.py` | Centraliza rutas, periodo de datos, versión y particiones. |
| `run_experiments.py` | Ejecuta los modelos baseline y registra sus resultados. |
| MLflow | Conserva el historial de experimentos, las métricas y los artefactos. |
| Git y pytest | Apoyan el control de versiones y la validación automatizada. |

**Nota:** P2 utiliza una etiqueta proxy de riesgo basada en los flujos y la capacidad registrada; no representa un inventario histórico real de bicicletas o anclajes.
