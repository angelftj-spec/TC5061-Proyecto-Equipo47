# Model Card – MiBici P1: Predicción de demanda

## 1. Información general

- **Proyecto:** MiBici – Equipo 47
- **Modelo:** mibici-equipo47-p1-demanda
- **Versión desplegada:** 1
- **Alias en MLflow Registry:** champion
- **Tipo de modelo:** RandomForestRegressor
- **Objetivo:** Predecir la cantidad de viajes del día siguiente.
- **Variable objetivo:** target_trips_next_day
- **Número de características:** 21
- **Run ID:** a5c48a44a54b4349a68610f1cc26fcb5

## 2. Uso previsto

El modelo está diseñado para estimar la demanda de viajes de MiBici por estación, utilizando información histórica y características derivadas de los datos disponibles.

Su propósito es apoyar el análisis y la planeación de la demanda del sistema.

No debe utilizarse como única fuente para tomar decisiones operativas críticas sin supervisión y validación adicional.

## 3. Datos de entrenamiento y evaluación

- **Periodo del conjunto de datos:** enero de 2023 a septiembre de 2026.
- **Entrenamiento:** enero de 2023 a diciembre de 2025.
- **Validación:** enero de 2026 a junio de 2026.
- **Prueba:** julio de 2026 a septiembre de 2026.
- **Registros de entrenamiento:** 518,562.
- **Registros de validación:** 87,966.
- **Registros de prueba:** 44,712.
- **Semilla aleatoria:** 42.

La versión del conjunto de datos y su hash SHA-256 están registrados en MLflow Tracking.

## 4. Arquitectura y configuración

Algoritmo: Random Forest Regressor.

Hiperparámetros registrados:

- n_estimators: 80
- max_depth: 14
- min_samples_leaf: 5
- random_state: 42

El modelo fue seleccionado por obtener el menor MAE de validación entre los candidatos P1 evaluados.

## 5. Resultados de evaluación

| Métrica | Validación | Prueba |
|---|---:|---:|
| MAE | 4.6565 | 5.6339 |
| RMSE | 10.0533 | 10.9310 |
| R² | 0.9011 | 0.9029 |

Los resultados muestran capacidad predictiva consistente entre los conjuntos de validación y prueba. El MAE representa el error absoluto medio en viajes.

## 6. Limitaciones y consideraciones

- El desempeño depende de la calidad y representatividad de los datos históricos.
- Los cambios en patrones de movilidad, disponibilidad de estaciones o comportamiento de usuarios pueden afectar las predicciones.
- El modelo puede presentar degradación ante cambios en la distribución de los datos.
- Las predicciones deben interpretarse como estimaciones, no como valores garantizados.
- Es necesario mantener el monitoreo de data drift y evaluar periódicamente el desempeño.

## 7. Monitoreo y gobernanza

El proyecto contempla:

- Monitoreo de cambios en la distribución de los datos mediante Evidently.
- Comparación offline de modelos mediante evaluación A/B.
- Seguimiento de experimentos, métricas, parámetros y artefactos mediante MLflow.
- Registro y versionado del modelo en MLflow Model Registry.
- Despliegue de inferencia mediante FastAPI y Docker.

La comparación A/B disponible es una evaluación offline; no representa una prueba con tráfico real de usuarios.

## 8. Linaje y reproducibilidad

- **MLflow Experiment ID:** 1
- **MLflow Run ID:** a5c48a44a54b4349a68610f1cc26fcb5
- **Git commit asociado al entrenamiento:** 57a1488991c022edb74c50ff9335014899118608
- **Repositorio:** https://github.com/angelftj-spec/TC5061-Proyecto-Equipo47

Los parámetros, métricas, versión de datos y artefactos del entrenamiento se encuentran registrados en MLflow.

## 9. Responsabilidad

Este modelo es una herramienta de apoyo analítico. Sus resultados deben utilizarse considerando sus limitaciones, la calidad de los datos y el contexto operativo. Se recomienda revisar periódicamente su desempeño y actualizarlo cuando se identifique degradación significativa.
