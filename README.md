# Proyecto MiBici — Equipo 47

Proyecto integrador de Machine Learning para analizar los viajes del sistema de bicicletas públicas MiBici de Guadalajara, Jalisco. Incluye ingesta reproducible de datos, ingeniería de variables, experimentos de regresión, clasificación y clustering, seguimiento con MLflow y registro de modelos.

## 1. Objetivos del proyecto

El equipo aborda tres problemas de Machine Learning.

### P1. Regresión: predicción de demanda

Predecir el número de viajes que iniciará una estación durante el día siguiente, utilizando información histórica de viajes y variables temporales.

- **Unidad de análisis:** estación y día.
- **Variable objetivo:** `target_trips_next_day`.
- **Métricas:** MAE, RMSE y R².

### P2. Clasificación: riesgo aproximado de disponibilidad

Identificar estaciones con posible presión de disponibilidad de bicicletas o anclajes, utilizando los flujos de entradas y salidas y la capacidad registrada de la estación.

- **Unidad de análisis:** estación y día.
- **Variable objetivo:** `target_risk_station_next_day`.
- **Métricas:** accuracy, balanced accuracy, precisión, recall, F1 y ROC-AUC, cuando estén disponibles.

**Limitación importante:** el conjunto de datos utilizado no contiene el inventario histórico real de bicicletas ni la ocupación de anclajes. Por tanto, la etiqueta de riesgo es una aproximación basada en el flujo neto de viajes y la capacidad registrada; no representa una observación real de una estación vacía o llena.

### P3. Clustering: agrupación de estaciones

Agrupar estaciones según sus patrones de uso y características geográficas para identificar perfiles de comportamiento.

- **Unidad de análisis:** estación.
- **Algoritmo:** K-Means.
- **Número de grupos:** se compararon varios valores de `k`.
- **Métricas:** silhouette score, inercia y tamaños de los grupos.

## 2. Fuente y versión de los datos

Fuente oficial: [Datos abiertos de MiBici](https://www.mibici.net/es/datos-abiertos/).

La ingesta utiliza los CSV mensuales publicados en el portal y la nomenclatura de estaciones.

| Elemento | Configuración |
|---|---|
| Inicio del periodo | Enero de 2023 |
| Fin del periodo | Septiembre de 2026 |
| Meses procesados | 45 |
| Viajes procesados | 16,833,133 |
| Estaciones en la nomenclatura actual | 484 |
| Formato procesado | Parquet |
| Versión de datos | `mibici-2023-01_to_2026-09-stations-2026-09` |

Los datos originales y procesados se conservan localmente y no se suben a GitHub. El archivo `references/data_manifest.json` contiene metadatos, conteos y hashes SHA-256 para identificar la versión de los datos.

La ingesta distingue los formatos de fecha ISO y día/mes/año para evitar asignar viajes al mes incorrecto. En la validación se comprobó que los 45 archivos procesados no tuvieran fechas vacías ni registros fuera del mes correspondiente.

Algunos identificadores históricos de estaciones pueden no estar presentes en la nomenclatura de septiembre de 2026. Los flujos históricos se conservan; para P3 se utilizan las estaciones de la nomenclatura actual que cuentan con coordenadas.

## 3. Partición temporal

Para evaluar predicciones futuras sin mezclar aleatoriamente observaciones del pasado y del futuro, se utiliza la fecha objetivo de cada muestra (`target_date`).

| Conjunto | Periodo |
|---|---|
| Entrenamiento | Enero de 2023 – diciembre de 2025 |
| Validación | Enero – junio de 2026 |
| Prueba | Julio – septiembre de 2026 |

La validación se utiliza para comparar modelos y ajustar hiperparámetros. La prueba se reserva para la evaluación final. En P1 y P2, el objetivo corresponde al día siguiente de la información histórica usada para generar las variables. El conjunto de prueba no debe utilizarse para seleccionar modelos ni ajustar hiperparámetros.

## 4. Estructura del repositorio

```text
TC5061-Proyecto-Equipo47/
├── data/                         # Datos locales, excluidos de Git
│   ├── raw/
│   ├── interim/
│   └── processed/
│       ├── trips/
│       ├── stations/
│       ├── features/
│       └── predictions/
├── references/
│   └── data_manifest.json
├── docs/
├── models/
├── notebooks/
├── reports/
│   ├── evidencias/
│   ├── figures/
│   ├── p1_candidate_results.csv
│   ├── p2_candidate_results.csv
│   └── p3_candidate_results.csv
├── mibici/
│   ├── config.py
│   ├── dataset.py
│   ├── features.py
│   └── modeling/
│       ├── train.py
│       ├── predict.py
│       ├── run_experiments.py
│       ├── run_candidates_p1.py
│       ├── run_candidates_p2.py
│       └── run_candidates_p3.py
├── tests/
├── MLproject
├── python_env.yaml
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── Makefile
└── README.md
```

La estructura exacta puede variar según los cambios integrados por el equipo. Los datos, el entorno virtual, las credenciales, la base local de MLflow y los artefactos locales están excluidos de Git mediante `.gitignore`.

## 5. Requisitos del entorno

El entorno de desarrollo verificado utiliza Python 3.14.7, MLflow 3.17.1, PyArrow 25.0.1, NumPy 2.5.4, pandas, scikit-learn, matplotlib y pytest. Las dependencias están fijadas en `requirements.txt`; `python_env.yaml` define el entorno usado por MLflow Projects.

Los siguientes comandos están escritos para Windows y PowerShell.

### 5.1. Clonar el repositorio

```powershell
git clone https://github.com/angelftj-spec/TC5061-Proyecto-Equipo47.git
cd TC5061-Proyecto-Equipo47
```

### 5.2. Crear e instalar el entorno

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Los comandos invocan explícitamente el Python del entorno `.venv`, por lo que no es necesario activar el entorno.

> **Nota sobre la reproducción:** los datos descargados, la base SQLite y los artefactos de MLflow no se incluyen en Git. La primera ejecución requiere acceso al portal de datos y generar localmente los archivos necesarios.

## 6. Reproducir la ingesta

El proyecto define un punto de entrada `ingest` en `MLproject`. Para ejecutar la ingesta mediante MLflow Projects desde la raíz del repositorio:

```powershell
.\.venv\Scripts\mlflow.exe run . --entry-point ingest --env-manager local
```

También se puede ejecutar directamente:

```powershell
.\.venv\Scripts\python.exe -m mibici.dataset
```

La ingesta utiliza el rango configurado en `mibici/config.py` y descarga los CSV que no existan localmente. Después:

1. Normaliza los nombres de las columnas y los formatos de fecha.
2. Procesa los viajes mensualmente.
3. Guarda los viajes y la nomenclatura en Parquet.
4. Registra conteos, rutas y hashes SHA-256 en `references/data_manifest.json`.

Para seleccionar un rango más pequeño, si se ejecuta el módulo directamente:

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

Las variables históricas de P1 y P2 incluyen rezagos y medias móviles calculadas con días anteriores. Se debe conservar esta separación temporal al incorporar nuevos predictores.

## 8. Experimentos reproducibles con MLflow Projects

La configuración `MLproject` define los puntos de entrada para la ingesta, entrenamiento y ejecución de los candidatos. El experimento de seguimiento se llama `mibici-equipo47`; la URI local de seguimiento utiliza SQLite.

### 8.1. Ejecutar los baselines

Ejecutar los tres problemas:

```powershell
.\.venv\Scripts\mlflow.exe run . --entry-point train --env-manager local
```

Ejecutar un problema específico:

```powershell
# P1: regresión
.\.venv\Scripts\mlflow.exe run . --entry-point train -P question=P1 --env-manager local

# P2: clasificación
.\.venv\Scripts\mlflow.exe run . --entry-point train -P question=P2 --env-manager local

# P3: clustering
.\.venv\Scripts\mlflow.exe run . --entry-point train -P question=P3 --env-manager local
```

Para especificar el número de grupos de P3:

```powershell
.\.venv\Scripts\mlflow.exe run . --entry-point train -P question=P3 -P n_clusters=4 --env-manager local
```

Los comandos suponen que los nombres de parámetros y los puntos de entrada coinciden con la versión de `MLproject` incluida en la rama. Si se modifica esa interfaz, debe actualizarse también esta documentación.

### 8.2. Ejecutar los candidatos

Los puntos de entrada para los scripts candidatos se definen en `MLproject`. Por ejemplo:

```powershell
.\.venv\Scripts\mlflow.exe run . --entry-point candidates_p1 --env-manager local
.\.venv\Scripts\mlflow.exe run . --entry-point candidates_p2 --env-manager local
.\.venv\Scripts\mlflow.exe run . --entry-point candidates_p3 --env-manager local
```

### 8.3. Baselines de referencia

| Pregunta | Modelo baseline | Métricas principales |
|---|---|---|
| P1 | `DummyRegressor(strategy="mean")` | MAE, RMSE, R² |
| P2 | `DummyClassifier(strategy="most_frequent")` | Accuracy, balanced accuracy, precisión, recall, F1 |
| P3 | `KMeans` con cuatro grupos | Silhouette score, inercia y tamaños de grupos |

Estos modelos establecen una referencia inicial y no representan necesariamente los modelos finales seleccionados. En P1, el baseline predice el promedio de los viajes del conjunto de entrenamiento. En P2, predice siempre la clase más frecuente y, en la ejecución inicial, no detectó las observaciones positivas de riesgo.

## 9. Comparación de modelos y resultados

Los CSV de resultados de candidatos se conservan en `reports/`. Las métricas de validación sirven para comparar candidatos; las métricas de prueba corresponden a la evaluación final y no deben mezclarse.

### 9.1. P1: predicción de demanda

Resultados de validación registrados para algunos candidatos:

| Modelo | MAE de validación |
|---|---:|
| Random Forest | 4.6565 |
| HistGradientBoosting | 4.8625 |
| Ridge | 6.4827 |

El modelo final registrado fue Random Forest. En el conjunto de prueba obtuvo MAE = 5.6339, RMSE = 10.9310 y R² = 0.9029. La diferencia entre validación y prueba debe interpretarse considerando que se trata de periodos temporales distintos.

### 9.2. P2: clasificación de riesgo

Resultados de validación de candidatos:

| Modelo | Accuracy | Balanced accuracy | Precisión | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.7272 | 0.7500 | 0.6233 | 0.8998 | 0.7365 | 0.8372 |
| HistGradientBoosting | 0.7438 | 0.7567 | 0.6534 | 0.8415 | 0.7356 | 0.8387 |
| Logistic Regression | 0.7108 | 0.6995 | 0.6702 | 0.6253 | 0.6470 | 0.7881 |

El modelo final registrado fue HistGradientBoosting. En el conjunto de prueba obtuvo F1 = 0.6773, recall = 0.8150 y ROC-AUC = 0.7665. Aunque Random Forest obtuvo un F1 de validación ligeramente superior en la tabla de candidatos, HistGradientBoosting presentó una ROC-AUC de validación ligeramente mayor y mejor balanced accuracy. La elección final debe explicarse en el informe con los criterios de selección usados por el equipo; no debe afirmarse que fue el mejor candidato en todas las métricas.

### 9.3. P3: agrupación de estaciones

| k | Silhouette score | Inercia | Tamaño mínimo | Tamaño máximo |
|---:|---:|---:|---:|---:|
| 3 | 0.1542 | 13976.42 | 13 | 242 |
| 4 | 0.1528 | 12556.96 | 13 | 215 |
| 5 | 0.1449 | 11742.50 | 8 | 189 |
| 6 | 0.1348 | 11173.05 | 8 | 135 |
| 8 | 0.1365 | 10312.64 | 1 | 136 |

El modelo final registrado fue K-Means con `k=3`, con silhouette score de 0.1542 y tamaños de grupo de 229, 242 y 13 estaciones. El silhouette score es moderado-bajo, por lo que los grupos deben interpretarse como perfiles descriptivos y no como categorías naturales inequívocas.

## 10. Consultar MLflow

Iniciar la interfaz desde la raíz del repositorio:

```powershell
.\.venv\Scripts\mlflow.exe ui `
    --backend-store-uri sqlite:///mlflow.db `
    --host 127.0.0.1 `
    --port 5000
```

Abrir [http://127.0.0.1:5000](http://127.0.0.1:5000).

El experimento `mibici-equipo47` registra parámetros, métricas, etiquetas y artefactos. La base `mlflow.db` y los artefactos permanecen localmente; no deben subirse a GitHub.

### 10.1. Model Registry

Los tres modelos finales se registraron en MLflow Model Registry con el alias `champion`:

| Problema | Nombre registrado |
|---|---|
| P1 | `mibici-equipo47-p1-demanda` |
| P2 | `mibici-equipo47-p2-riesgo` |
| P3 | `mibici-equipo47-p3-grupos` |

El script de inferencia resuelve el alias `champion` y carga el modelo registrado. Para que funcione, deben existir localmente la base de MLflow, los modelos registrados y los archivos de características procesadas.

## 11. Ejecutar inferencia

Después de completar la ingesta, generar las características y registrar los tres modelos finales, ejecutar desde la raíz del repositorio:

```powershell
.\.venv\Scripts\python.exe -m mibici.modeling.predict
```

El programa valida las columnas requeridas, carga los modelos registrados y genera:

| Archivo | Descripción |
|---|---|
| `data/processed/predictions/p1_demand_predictions.csv` | Predicciones de demanda |
| `data/processed/predictions/p2_risk_predictions.csv` | Clasificación aproximada del riesgo por estación |
| `data/processed/predictions/p3_station_clusters.csv` | Asignación de estaciones a grupos |

Las predicciones son resultados del modelo; no constituyen garantías sobre la demanda ni observaciones reales de disponibilidad de bicicletas.

## 12. Pruebas automatizadas

Ejecutar las pruebas:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Las pruebas actuales verifican la normalización de nombres, el mapeo de columnas de estaciones, la interpretación de fechas y el rango mensual. La ingesta también debe validarse mediante los hashes del manifiesto y las fechas de los Parquet.

## 13. Flujo de trabajo con GitHub

Los cambios deben integrarse mediante Pull Requests dirigidos a `main`, siguiendo las reglas de protección y revisión del repositorio.

Ejemplo de flujo de trabajo:

```powershell
git switch main
git pull origin main
git switch -c feat/descripcion-del-cambio
```

Después de implementar y probar los cambios:

```powershell
git status
git add README.md
git commit -m "Document reproducible MiBici workflow"
git push -u origin feat/descripcion-del-cambio
```

Abrir un Pull Request hacia `main` y solicitar la revisión de otro integrante. No subir los CSV originales, los Parquet, `.env`, `.venv`, la base local de MLflow ni los modelos serializados generados durante los experimentos.

## 14. Evidencias del proyecto

Las capturas de seguimiento y artefactos se conservan en `reports/evidencias/`, y las tablas de candidatos en `reports/`. Entre las evidencias previstas están la vista de runs, las comparaciones de experimentos, el detalle de una ejecución, los artefactos de evaluación de cada problema y los modelos registrados.

Las rutas de evidencia deben corresponder a los archivos realmente presentes en el repositorio. Las evidencias visuales complementan, pero no sustituyen, las métricas y los artefactos reproducibles de MLflow.

## 15. Limitaciones y próximos pasos

- La etiqueta de riesgo de P2 es una aproximación; no representa observaciones históricas reales de estaciones vacías o llenas.
- Los grupos de P3 son descriptivos y dependen de las variables utilizadas, incluidas las coordenadas geográficas.
- Las métricas de validación y prueba corresponden a periodos diferentes y deben reportarse por separado.
- La selección final de P2 debe justificarse con los criterios usados por el equipo, dado que Random Forest tuvo un F1 de validación ligeramente mayor que HistGradientBoosting.
- Debe comprobarse que las instrucciones de reproducción funcionen desde un clon limpio con las dependencias y los datos locales necesarios.

## Referencias

MiBici. (s. f.). *Datos abiertos*. https://www.mibici.net/es/datos-abiertos/

MLflow. (s. f.). *MLflow documentation*. https://mlflow.org/docs/latest/

scikit-learn developers. (s. f.). *scikit-learn documentation*. https://scikit-learn.org/stable/
