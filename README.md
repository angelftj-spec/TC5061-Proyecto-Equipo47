# Proyecto MiBici — Equipo 47

Proyecto integrador de Machine Learning para analizar los viajes del sistema de bicicletas públicas MiBici de Guadalajara, Jalisco. El proyecto incluye ingesta reproducible de datos, ingeniería de variables, experimentos de regresión, clasificación y clustering, y seguimiento de experimentos con MLflow.

## 1. Objetivos del proyecto

El equipo aborda tres problemas de Machine Learning:

### P1. Regresión: predicción de demanda

Predecir el número de viajes que iniciará una estación durante el día siguiente, utilizando información histórica de viajes y variables temporales.

- Unidad de análisis: estación y día.
- Variable objetivo: `target_trips_next_day`.
- Métricas: MAE, RMSE y R².

### P2. Clasificación: riesgo aproximado de disponibilidad

Identificar estaciones con posible presión de disponibilidad de bicicletas o anclajes, utilizando los flujos de entradas y salidas y la capacidad registrada de la estación.

- Unidad de análisis: estación y día.
- Variable objetivo: `target_risk_station_next_day`.
- Métricas: accuracy, balanced accuracy, precisión, recall y F1.

**Limitación importante:** MiBici no proporciona en este conjunto de datos el inventario histórico real de bicicletas ni la ocupación de anclajes. Por tanto, la etiqueta de riesgo es una aproximación basada en el flujo neto de viajes y la capacidad registrada, no una observación real de una estación vacía o llena.

### P3. Clustering: agrupación de estaciones

Agrupar estaciones según sus patrones de uso y características geográficas para identificar perfiles de comportamiento.

- Unidad de análisis: estación.
- Algoritmo baseline: K-Means.
- Número inicial de grupos: `k = 4`.
- Métrica inicial: silhouette score.

## 2. Fuente y versión de los datos

Fuente oficial: [Datos abiertos de MiBici](https://www.mibici.net/es/datos-abiertos/).

La ingesta utiliza los CSV mensuales publicados en el portal y la nomenclatura de estaciones.

| Elemento | Configuración |
|---|---|
| Inicio del periodo | Enero de 2023 |
| Fin del periodo | Septiembre de 2026 |
| Meses procesados | 45 |
| Viajes procesados | 16,833,133 |
| Estaciones en nomenclatura actual | 484 |
| Formato procesado | Parquet |
| Versión de datos | `mibici-2023-01_to_2026-09-stations-2026-09` |

Los datos originales y procesados se conservan localmente y no se suben a GitHub. El archivo `references/data_manifest.json` contiene los metadatos, los conteos y los hashes SHA-256 utilizados para identificar la versión de los datos.

La ingesta distingue los formatos de fecha ISO y día/mes/año para evitar asignar viajes al mes incorrecto. Durante la validación se comprobó que los 45 archivos procesados no tuvieran fechas vacías ni registros fuera del mes correspondiente.

Algunos identificadores históricos de estaciones pueden no estar presentes en la nomenclatura de septiembre de 2026. Los flujos históricos se conservan; para P3 se utilizan las estaciones de la nomenclatura actual que cuentan con coordenadas.

## 3. Partición temporal

Para evaluar predicciones futuras sin mezclar aleatoriamente observaciones del pasado y del futuro, se utiliza la fecha objetivo de cada muestra (`target_date`).

| Conjunto | Periodo |
|---|---|
| Entrenamiento | Enero de 2023 – diciembre de 2025 |
| Validación | Enero – junio de 2026 |
| Prueba | Julio – septiembre de 2026 |

La validación se utiliza para comparar modelos y ajustar hiperparámetros. La prueba se reserva para la evaluación final.

En P1 y P2, el objetivo corresponde al día siguiente de la información histórica usada para generar las variables. El conjunto de prueba no debe utilizarse para seleccionar modelos ni ajustar hiperparámetros.

## 4. Estructura del repositorio

```text
TC5061-Proyecto-Equipo47/
├── data/
│   ├── raw/
│   │   ├── trips/
│   │   └── stations/
│   ├── interim/
│   └── processed/
│       ├── trips/
│       ├── stations/
│       └── features/
├── references/
│   └── data_manifest.json
├── models/
├── notebooks/
├── reports/
│   └── figures/
├── mibici/
│   ├── config.py
│   ├── dataset.py
│   ├── features.py
│   └── modeling/
│       ├── train.py
│       ├── predict.py
│       └── run_experiments.py
├── tests/
│   └── test_data.py
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── Makefile
└── README.md
```

Los archivos de datos, el entorno virtual, las credenciales y el almacenamiento local de MLflow están excluidos de Git mediante `.gitignore`.

## 5. Requisitos del entorno

El entorno de desarrollo verificado utiliza:

- Python 3.14.7
- MLflow 3.17.1
- PyArrow 25.0.1
- NumPy, pandas, scikit-learn, matplotlib y pytest

Las dependencias están fijadas en `requirements.txt`. Para facilitar la reproducción, se recomienda utilizar la versión de Python y las dependencias indicadas en el repositorio.

Los comandos siguientes están escritos para Windows y PowerShell.

### 5.1. Clonar el repositorio

```powershell
git clone https://github.com/angelftj-spec/TC5061-Proyecto-Equipo47.git
cd TC5061-Proyecto-Equipo47
```

### 5.2. Crear el entorno virtual

```powershell
py -3.14 -m venv .venv
```

Instalar las dependencias:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

En los siguientes pasos se invoca explícitamente el Python del entorno `.venv`, por lo que no es necesario activar el entorno para ejecutar los comandos.

## 6. Reproducir la ingesta de datos

Desde la raíz del repositorio, ejecutar:

```powershell
.\.venv\Scripts\python.exe -m mibici.dataset
```

El comando utiliza el rango configurado en `mibici/config.py` y descarga los CSV que no existan localmente. Después:

1. Normaliza los nombres de las columnas y los formatos de fecha.
2. Procesa los viajes mensualmente.
3. Guarda los viajes y la nomenclatura en Parquet.
4. Registra conteos, rutas y hashes SHA-256 en `references/data_manifest.json`.

Para seleccionar un rango más pequeño:

```powershell
.\.venv\Scripts\python.exe -m mibici.dataset `
    --start-month 2023-01 `
    --end-month 2023-03 `
    --station-month 2026-09
```

Para forzar la descarga de los archivos originales:

```powershell
.\.venv\Scripts\python.exe -m mibici.dataset --force-download
```

## 7. Generar las variables

Después de completar la ingesta, ejecutar:

```powershell
.\.venv\Scripts\python.exe -m mibici.features
```

El comando genera los siguientes archivos:

| Archivo | Contenido |
|---|---|
| `data/processed/features/daily_station_flows.parquet` | Viajes iniciados y terminados por estación y día |
| `data/processed/features/p1_regression.parquet` | Variables históricas y objetivo de regresión |
| `data/processed/features/p2_classification.parquet` | Variables y etiquetas aproximadas de riesgo |
| `data/processed/features/p3_clustering.parquet` | Perfiles de uso por estación |

Las variables históricas de P1 y P2 incluyen rezagos y medias móviles calculadas con días anteriores. El código debe mantener esta separación temporal al incorporar nuevos predictores.

## 8. Ejecutar los experimentos baseline

Los experimentos baseline se registran en el experimento MLflow `mibici-equipo47`.

Ejecutar los tres experimentos:

```powershell
.\.venv\Scripts\python.exe -m mibici.modeling.run_experiments
```

Ejecutar solo una pregunta:

```powershell
# P1: regresión
.\.venv\Scripts\python.exe -m mibici.modeling.run_experiments --question P1

# P2: clasificación
.\.venv\Scripts\python.exe -m mibici.modeling.run_experiments --question P2

# P3: clustering
.\.venv\Scripts\python.exe -m mibici.modeling.run_experiments --question P3
```

Modificar el número de grupos de P3:

```powershell
.\.venv\Scripts\python.exe -m mibici.modeling.run_experiments --question P3 --n-clusters 4
```

### 8.1. Baselines actuales

| Pregunta | Modelo baseline | Métricas |
|---|---|---|
| P1 | `DummyRegressor(strategy="mean")` | MAE, RMSE, R² |
| P2 | `DummyClassifier(strategy="most_frequent")` | Accuracy, balanced accuracy, precisión, recall, F1 |
| P3 | `KMeans` con cuatro grupos | Silhouette score, inercia y tamaños de grupos |

Estos modelos establecen una referencia inicial y **no representan los modelos finales seleccionados**.

En P1, el baseline predice el promedio de los viajes del conjunto de entrenamiento. En P2, el baseline predice siempre la clase más frecuente y, en la ejecución inicial, no detectó las observaciones positivas de riesgo. En P3, los perfiles numéricos se imputan con la mediana y se escalan antes de ejecutar K-Means.

Los modelos candidatos de los integrantes deben evaluarse con las mismas particiones temporales y compararse con sus respectivos baselines.

## 9. Consultar MLflow

El seguimiento de los experimentos utiliza SQLite en `mlflow.db`. Los artefactos y los resultados de MLflow permanecen localmente.

Iniciar la interfaz desde la raíz del proyecto:

```powershell
.\.venv\Scripts\mlflow.exe ui `
    --backend-store-uri sqlite:///mlflow.db `
    --host 127.0.0.1 `
    --port 5000
```

Abrir en el navegador:

[http://127.0.0.1:5000](http://127.0.0.1:5000)

El experimento `mibici-equipo47` registra parámetros, métricas, etiquetas y artefactos. Cada run incluye metadatos de la versión de datos y del commit de Git.

Entre los artefactos de evaluación se encuentran las gráficas y las muestras de predicciones, así como el mapa y las asignaciones de clusters.

No subir `mlflow.db`, `mlruns/` ni `mlartifacts/` al repositorio.

## 10. Pruebas automatizadas

Ejecutar las pruebas:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Las pruebas actuales verifican la normalización de nombres, el mapeo de columnas de estaciones, la interpretación de fechas y el rango mensual.

La ingesta también debe validarse mediante los hashes del manifiesto y las fechas de los Parquet. Los cambios al pipeline deben conservar estas verificaciones.

## 11. Flujo de trabajo con GitHub

La rama `main` está protegida. Los cambios deben integrarse mediante Pull Requests y recibir al menos una aprobación.

Después de que la base del proyecto esté integrada en `main`, cada integrante debe crear una rama independiente:

```powershell
git switch main
git pull origin main
git switch -c feat/p1-model-iniciales
```

Sustituir `p1-model-iniciales` por la pregunta y las iniciales del integrante. Usar commits descriptivos y abrir un Pull Request dirigido a `main` para revisión.

No se deben subir los CSV originales, los Parquet, `.env`, `.venv`, la base local de MLflow ni los modelos serializados generados durante los experimentos.

## 12. Estado actual y próximos pasos

### Trabajo realizado

- Implementación y ejecución de experimentos baseline para P1, P2 y P3.
- Entrenamiento y comparación de modelos candidatos mediante los scripts de modelado.
- Selección de modelos finales con base en los criterios de evaluación establecidos.
- Registro de experimentos, métricas y artefactos en MLflow.
- Registro de los resultados de candidatos y de evidencias visuales en `reports/`.
- Entrenamiento y registro de los modelos finales para las tres preguntas.
- Preparación de los cambios en la rama `feat/mlflow-project-advanced` y apertura de un Pull Request dirigido a `main`.

### Pendientes

- Revisar los resultados y la documentación con el equipo.
- Verificar que el informe final explique los criterios de selección, las métricas y las limitaciones de cada modelo.
- Confirmar que las evidencias del repositorio sean suficientes para los requisitos de la entrega.
- Atender los comentarios y las solicitudes de cambios del Pull Request.
- Integrar los cambios en `main` después de obtener la aprobación requerida.

### Consideraciones

- P1 y P2 deben evaluarse respetando la separación cronológica entre entrenamiento, validación y prueba.
- La etiqueta de riesgo de P2 es una aproximación; no representa observaciones históricas reales de estaciones vacías o llenas.
- Los grupos de P3 son descriptivos y dependen de las variables utilizadas, incluidas las coordenadas geográficas.
- La base de datos local de MLflow, los datos originales, los Parquet y los modelos serializados deben permanecer fuera de GitHub.
