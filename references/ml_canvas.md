
# ML Canvas — Proyecto MiBici, Equipo 47

## 1. Problema y propuesta de valor

El sistema de bicicletas públicas MiBici necesita información que ayude a comprender la demanda de viajes, anticipar posibles desequilibrios de disponibilidad e identificar estaciones con patrones de uso similares.

El proyecto propone tres soluciones de Machine Learning sobre los viajes históricos y la nomenclatura de estaciones. Los resultados servirán como apoyo analítico para la planeación operativa y urbana; no sustituyen los sistemas de inventario ni las decisiones del operador.

## 2. Usuarios y decisiones

| Usuario | Necesidad | Decisión apoyada |
|---|---|---|
| Operación de MiBici | Anticipar la demanda por estación | Planear la distribución de bicicletas y los recursos operativos |
| Equipo de rebalanceo | Identificar estaciones con posibles desequilibrios | Priorizar inspecciones y movimientos de bicicletas |
| Planeación urbana | Comprender perfiles de uso y su distribución geográfica | Investigar estaciones que podrían requerir ajustes o expansión del sistema |

## 3. Formulación de las preguntas de ML

### P1. Regresión: predicción de demanda

**Pregunta de negocio:** ¿Cuántos viajes se iniciarán en una estación durante un periodo futuro?

- **Predicción:** número de viajes iniciados por estación durante el día siguiente.
- **Unidad de análisis:** estación-día.
- **Horizonte:** un día adelante.
- **Variable objetivo:** `target_trips_next_day`.
- **Datos de entrada:** viajes históricos, estación de origen, variables de calendario, rezagos de demanda y medias móviles calculadas con información pasada.
- **Usuario:** operador del sistema.
- **Decisión:** utilizar las estimaciones para apoyar la planeación de la flota y anticipar estaciones con demanda elevada.
- **Baseline:** `DummyRegressor(strategy="mean")`.
- **Modelos candidatos:** regresión lineal y Random Forest Regressor.
- **Métricas:** MAE, RMSE y R².

La métrica principal propuesta es MAE, porque expresa el error medio en número de viajes y facilita su interpretación operativa. RMSE penaliza más los errores grandes y R² complementa el análisis sobre la variabilidad explicada.

### P2. Clasificación: riesgo aproximado de disponibilidad

**Pregunta de negocio:** ¿Una estación estará en riesgo de quedarse sin bicicletas o sin anclajes disponibles?

- **Predicción:** etiqueta binaria de riesgo aproximado para el día siguiente.
- **Unidad de análisis:** estación-día.
- **Horizonte:** un día adelante.
- **Variable objetivo principal:** `target_risk_station_next_day`.
- **Datos de entrada:** flujos históricos de llegadas y salidas, rezagos de demanda, medias móviles y capacidad registrada de la estación.
- **Usuario:** equipo de rebalanceo.
- **Decisión:** priorizar estaciones para revisión operativa y posible rebalanceo.
- **Baseline:** `DummyClassifier(strategy="most_frequent")`.
- **Modelos candidatos:** Logistic Regression, Random Forest Classifier y HistGradientBoosting. El modelo final seleccionado fue HistGradientBoosting.
- **Métricas:** precisión, recall, F1, balanced accuracy y matriz de confusión.

El recall de la clase de riesgo será importante porque los falsos negativos pueden ocultar estaciones que merecen atención. La precisión ayudará a evaluar cuántas alertas generadas corresponden a casos positivos según la etiqueta utilizada.

**Limitación de la variable objetivo:** el conjunto de datos no contiene inventarios históricos reales de bicicletas ni ocupación de anclajes. La etiqueta implementada es una aproximación: compara el desequilibrio neto de llegadas y salidas del día siguiente con el 20% de la capacidad registrada. No representa una observación real de una estación vacía o llena. Esta hipótesis deberá evaluarse y documentarse antes de interpretar sus resultados operativamente.

### P3. Clustering: perfiles de uso de estaciones

**Pregunta de negocio:** ¿Qué grupos de estaciones comparten patrones de uso y dónde se ubican en la ciudad?

- **Resultado:** asignación de cada estación a un grupo de comportamiento.
- **Unidad de análisis:** estación.
- **Horizonte:** análisis descriptivo del periodo observado, sin predecir un valor futuro.
- **Datos de entrada:** promedios de llegadas y salidas diarias, flujo neto, actividad en días laborales y fines de semana, proporciones horarias, capacidad y coordenadas geográficas.
- **Usuario:** equipo de planeación urbana.
- **Decisión:** comparar perfiles de uso y estudiar la distribución espacial de las estaciones.
- **Baseline:** K-Means con cuatro grupos. Para el modelo final se utilizó K-Means con tres grupos; ambos resultados deben distinguirse al comparar la evaluación inicial con la selección final.
- **Métricas:** silhouette score e inercia, complementadas con el tamaño y la interpretación de cada grupo.

Las coordenadas geográficas forman parte de las variables utilizadas por el baseline, por lo que la similitud entre estaciones depende tanto de sus características de uso como de su posición geográfica. Se deberá considerar esta decisión al interpretar los grupos.

## 4. Datos y preparación

- **Fuente:** portal oficial de datos abiertos de MiBici.
- **Periodo:** enero de 2023 a septiembre de 2026.
- **Volumen procesado:** 16,833,133 viajes en 45 archivos mensuales.
- **Nomenclatura:** 484 estaciones en el catálogo de septiembre de 2026.
- **Formato procesado:** Parquet.
- **Versión de datos:** `mibici-2023-01_to_2026-09-stations-2026-09`.
- **Reproducibilidad:** manifiesto con conteos y hashes SHA-256.

Se normalizan los encabezados, los tipos de fecha y los identificadores de estación. Las variables utilizadas para las predicciones deben estar disponibles en el momento de generar la predicción; no deben incorporar información posterior al horizonte objetivo.

La partición de P1 y P2 se establece cronológicamente según `target_date`:

- Entrenamiento: enero de 2023 a diciembre de 2025.
- Validación: enero a junio de 2026.
- Prueba: julio a septiembre de 2026.

El conjunto de prueba debe reservarse para la evaluación final, sin utilizarlo para seleccionar modelos ni ajustar hiperparámetros. P3 es un análisis descriptivo sobre el periodo completo observado y no utiliza esta misma partición supervisada.

## 5. Infraestructura y seguimiento

- Ingesta reproducible: `mibici/dataset.py`.
- Configuración central: `mibici/config.py`.
- Ingeniería de variables: `mibici/features.py`.
- Ejecutor baseline: `mibici/modeling/run_experiments.py`.
- Seguimiento experimental: MLflow, con el experimento `mibici-equipo47`.
- Almacenamiento de seguimiento local: SQLite.
- Control de versiones de código: Git y GitHub.
- Pruebas automatizadas: pytest.

Cada run registra parámetros, métricas, etiquetas que identifican la pregunta y el equipo, referencia a la versión de datos, commit de Git y artefactos de evaluación.

## 6. Privacidad y condiciones de uso

Los datos se obtienen del portal público de MiBici, que ofrece descargas mensuales en CSV y un enlace al manual para uso de datos abiertos. Antes de redistribuir datos o afirmar que están sujetos a una licencia específica, se deben revisar las condiciones del manual; la página de descarga por sí sola no permite afirmar una licencia permisiva concreta.

El aviso de privacidad de MiBici describe el tratamiento de datos personales por la entidad responsable. Para este proyecto, el procesamiento conserva las fechas de inicio y fin, los identificadores de estación y la duración derivada del viaje; excluye del conjunto procesado los identificadores de usuario y variables demográficas presentes en los archivos de origen.

Los CSV, los Parquet, las credenciales y el almacenamiento local de MLflow permanecen fuera de GitHub. Las salidas compartidas deben limitarse a métricas agregadas, gráficas y muestras de evaluación que no revelen información personal.

## 7. Criterios de éxito

1. Cada pregunta tiene un baseline reproducible registrado en MLflow.
2. Las métricas permiten comparar modelos candidatos con su baseline correspondiente.
3. La evaluación supervisada respeta la partición temporal.
4. Los resultados y las versiones de datos pueden rastrearse mediante el manifiesto y los metadatos de Git.
5. Los grupos de P3 pueden interpretarse mediante sus perfiles de uso y su representación geográfica.
6. Las limitaciones de la etiqueta proxy de P2 están documentadas y no se presentan como inventario real.

## 8. Referencias principales

- MiBici. Datos abiertos: https://www.mibici.net/es/datos-abiertos/
- MiBici. Aviso de privacidad: https://www.mibici.net/es/aviso-de-privacidad/
- MLflow. Tracking: https://mlflow.org/docs/latest/ml/tracking/
- Scikit-learn. Model evaluation: https://scikit-learn.org/stable/modules/model_evaluation.html
