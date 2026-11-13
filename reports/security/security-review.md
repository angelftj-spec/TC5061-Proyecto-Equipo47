# Security Review — MiBici Equipo 47

**Fecha:** 2026-11-01
**Rama:** feat/diego-security-audit
**Entorno:** Windows, Python 3.11.9

## 1. Objetivo

Realizar una revisión básica de seguridad del repositorio,
incluyendo protección de archivos sensibles, auditoría de
dependencias, análisis estático del código y mitigación de
riesgos identificados.

## 2. Herramientas utilizadas

- Git y .gitignore
- pip-audit 2.10.1
- Bandit 1.9.4
- PyTest 9.1.1
- Python 3.11.9

## 3. Protección de información sensible

Se revisaron los archivos rastreados por Git en busca de
archivos .env, bases de datos y extensiones asociadas a llaves
privadas.

No se encontraron coincidencias con los patrones evaluados.

Se comprobó que .gitignore excluye archivos locales como
.env, .venv, data/ y mlflow.db.

Se agregaron reglas de exclusión para variantes de .env y
archivos .pem, .key, .p12 y .pfx.

También se buscaron términos relacionados con credenciales
en archivos Python, YAML y TOML, sin coincidencias.

Estos resultados no garantizan la ausencia absoluta de
secretos ni sustituyen una auditoría del historial de Git.

## 4. Auditoría de dependencias

Herramienta: pip-audit 2.10.1

Comando:

    python -m pip_audit --local

Resultado inicial:
- 59 registros de vulnerabilidad reportados.
- 7 paquetes afectados.
- El paquete local mibici no pudo ser auditado en PyPI.

Dependencias afectadas:
- certifi 2026.7.22
- GitPython 3.2.0
- MLflow 3.17.1
- Pillow 12.3.0
- pyasn1 0.6.4
- setuptools 65.5.0
- Tornado 6.5.10

Las vulnerabilidades siguen pendientes de evaluación
individual y remediación.

No se actualizaron automáticamente las dependencias para
evitar incompatibilidades con DeepChecks y los modelos.

## 5. Análisis estático con Bandit

Herramienta: Bandit 1.9.4

Comando:

    python -m bandit -r mibici

Resultados iniciales:
- 9 hallazgos en 3,040 líneas de código.
- Severidad alta: 0
- Severidad media: 2
- Severidad baja: 7

Resultados posteriores a la corrección:
- 7 hallazgos en 3,072 líneas de código.
- Severidad alta: 0
- Severidad media: 0
- Severidad baja: 7

Los dos hallazgos B310 de severidad media se eliminaron
mediante validaciones explícitas de URLs de descarga.

## 6. Correcciones implementadas

### Validación de URLs

Se incorporó una función de validación que permite
únicamente URLs HTTPS de los dominios autorizados de MiBici.

También se implementó una comprobación de redirecciones
HTTP para rechazar destinos externos no autorizados.

### Pruebas de seguridad

Se agregaron pruebas automatizadas para verificar:

- Aceptación de URLs HTTPS oficiales.
- Rechazo de URLs HTTP.
- Rechazo de esquemas file://.
- Rechazo de dominios externos.
- Rechazo de dominios que intentan suplantar MiBici.
- Rechazo de redirecciones externas.

El conjunto completo pasó de 14 a 20 casos de prueba.

### Configuración del entorno

Se alineó python_env.yaml con Python 3.11.9,
versión utilizada por el equipo después de integrar
DeepChecks.

## 7. Validación posterior a las correcciones

PyTest:
- 20 pruebas aprobadas.
- 0 pruebas fallidas.
- 1 FutureWarning de scikit-learn.

Bandit:
- 0 hallazgos de severidad media.
- 0 hallazgos de severidad alta.
- 7 hallazgos de severidad baja.

Verificación funcional:
- Consulta exitosa al portal oficial de MiBici.
- 140 enlaces mensuales identificados.
- Enlace de septiembre de 2026 encontrado correctamente.

La consulta exitosa confirma el acceso al índice,
pero no equivale a validar todas las descargas.

## 8. Riesgos pendientes y recomendaciones

### Dependencias

Evaluar las vulnerabilidades reportadas por pip-audit
y probar versiones corregidas en un entorno separado.

### Subprocess

Los siete hallazgos de severidad baja corresponden
a importaciones y llamadas a subprocess para obtener
metadatos de Git.

Las llamadas utilizan listas de argumentos fijos y
no emplean shell=True, lo que limita el riesgo
de inyección mediante argumentos.

Sin embargo, el ejecutable git se resuelve mediante
PATH. Se recomienda revisar este riesgo antes de
utilizar el código en ambientes no confiables.

### Alcance

La revisión no incluye pruebas de penetración,
análisis completo del historial de Git ni una
evaluación integral de todas las dependencias.

## 9. Evidencias

- reports/security/pip-audit.json
- reports/security/bandit.json
- Capturas de PyTest, pip-audit y Bandit en images/

Los reportes reflejan los resultados de las
herramientas en el momento de su ejecución.
