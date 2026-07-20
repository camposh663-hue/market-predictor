# Market Predictor — Estado del proyecto y hoja de ruta

> Este documento está pensado para que cualquier persona, aunque no haya visto el proyecto nunca, entienda qué es, cómo está construido, qué hay hecho hasta ahora y qué falta por hacer.

---

## 1. ¿Qué es Market Predictor?

Market Predictor **no es** un bot de trading, ni un modelo específico para Bitcoin, ni una colección de notebooks sueltos.

Es una **plataforma de Machine Learning** diseñada para:

- Entrenar modelos que predigan la **variación porcentual futura del precio** de cualquier instrumento financiero (BTC/USDT, ETH/USDT, AAPL, TSLA, EUR/USD, S&P500...).
- Predecir esa variación sobre **horizontes configurables** (30 minutos, 1 hora, 4 horas, 24 horas...).
- Comparar distintos algoritmos (Random Forest, XGBoost, LightGBM, CatBoost, LSTM, Transformers...) sin que ninguno esté "quemado" en el código.
- Funcionar con **cualquier proveedor de datos** (Binance, Yahoo Finance, Polygon...) sin depender de ninguno en particular.

Se construye como si fuera software de producción, no como un proyecto universitario: prioriza mantenibilidad, modularidad y arquitectura limpia por encima de ir rápido.

---

## 2. Filosofía de arquitectura (Clean Architecture)

La regla de oro del proyecto es: **cada módulo tiene una única responsabilidad**, y las capas internas (dominio) nunca dependen de las capas externas (proveedores, storage, frameworks).

```
Dominio  ←  Providers  ←  Database  ←  Sync  ←  (futuro: Features, Training, Prediction, App)
```

- El **dominio** define el lenguaje del negocio (qué es un instrumento, qué es una vela de precio) y no sabe que existen Binance, Parquet o pandas.
- Las capas externas (**providers**, **database**) traducen el mundo exterior (APIs, archivos) al lenguaje del dominio, nunca al revés.
- Las capas de orquestación (**sync**, y en el futuro **training**, **prediction**) combinan piezas del dominio y de las capas externas, pero sin contener ellas mismas reglas de negocio de bajo nivel.

Este patrón se repite una y otra vez en el proyecto: una interfaz abstracta (contrato) + implementaciones concretas intercambiables. Ejemplos: `BaseProvider` → `BinanceProvider`; `BarRepository` → `InMemoryRepository` / `ParquetRepository`.

---

## 3. Glosario de conceptos clave

| Concepto | Qué es |
|---|---|
| **Instrument** | Un instrumento financiero (ej. `BTC/USDT`, `AAPL`), descrito de forma independiente del proveedor que lo sirva. |
| **MarketBar** | Una "vela" OHLCV normalizada: `timestamp`, `open`, `high`, `low`, `close`, `volume`. |
| **TimeFrame** | El intervalo de agregación de una vela: `1m`, `5m`, `15m`, `1h`, `4h`, `1d`. |
| **AssetClass** | La categoría del instrumento: `crypto`, `equity`, `forex`, `index`, `other`. |
| **Provider** | Componente que descarga datos de una fuente externa (Binance, Yahoo...) y los traduce a `MarketBar`. |
| **Repository** | Componente que guarda y lee `MarketBar` de un almacenamiento concreto (memoria, Parquet, base de datos...). |
| **DataManager** | Orquestador que decide qué guardar y cómo (deduplicar, ordenar), delegando el "cómo se guarda" al Repository. |
| **Sync Service** | Conecta un Provider con un DataManager: decide qué rango de tiempo falta y lo descarga. |
| **Parquet** | Formato de archivo columnar, compacto y rápido para datos analíticos. Es el formato de almacenamiento elegido para el histórico de precios. |

---

## 4. Lo que existe hoy, capa por capa

### 4.1. Capa de dominio — `src/domain/`

El lenguaje del negocio. No importa nada de fuera del propio dominio. Es inmutable (`@dataclass(frozen=True)`) para evitar bugs por mutación accidental.

- **`asset_class.py`** — `AssetClass`: enum con las categorías de instrumento (`CRYPTO`, `EQUITY`, `FOREX`, `INDEX`, `OTHER`).
- **`timeframe.py`** — `TimeFrame`: enum con los intervalos soportados (`1m`, `5m`, `15m`, `1h`, `4h`, `1d`). El resto del código nunca usa strings sueltos como `"1h"`, siempre este enum.
- **`instrument.py`** — `Instrument`: describe un instrumento (`symbol`, `asset_class`, y opcionalmente `venue`, `base_currency`, `quote_currency`) sin atarse a ningún proveedor concreto.
- **`market_bar.py`** — `MarketBar`: una vela OHLCV normalizada (`timestamp`, `open`, `high`, `low`, `close`, `volume`). Es el objeto que viaja por todo el pipeline: de un provider a un repository, de un repository a un dataset, etc.

**Para qué sirve:** es el vocabulario común que entienden todas las demás capas. Si mañana cambiamos de proveedor o de storage, el dominio no se toca.

### 4.2. Capa de providers — `src/providers/`

Responsables **únicamente** de traer datos de una fuente externa y convertirlos a objetos del dominio.

- **`base_provider.py`** — `BaseProvider`: contrato abstracto. Define `provider_id` (identificador estable, ej. `"binance"`) y `get_historical_bars(instrument, timeframe, start, end)`, que debe devolver una lista de `MarketBar` ordenada cronológicamente.
- **`binance_provider.py`** — `BinanceProvider`: implementación concreta que habla con la API pública de Binance Spot, traduce símbolos e intervalos al formato de Binance, pagina las peticiones y convierte cada *kline* de Binance en un `MarketBar`.

**Para qué sirve:** el resto de la aplicación nunca ve JSON de Binance ni sabe cómo pedirle datos — solo conoce `BaseProvider`. Añadir un proveedor nuevo (Yahoo, Polygon) significa escribir una clase nueva que cumpla este contrato, sin tocar nada más.

### 4.3. Capa de almacenamiento — `src/database/`

Responsable de guardar, leer y actualizar `MarketBar`. **Nunca** habla con providers.

- **`base_repository.py`** — `BarRepository`: contrato abstracto de persistencia pura (I/O). Define `write_bars`, `read_bars` y `latest_timestamp`. No contiene reglas de negocio, solo lectura/escritura.
- **`in_memory_repository.py`** — `InMemoryRepository`: implementación en memoria (un diccionario). Se usa en tests para no depender del disco ni de un formato concreto.
- **`parquet_repository.py`** — `ParquetRepository`: implementación real sobre archivos Parquet en disco. Guarda cada serie (instrumento + timeframe) en su propio archivo, con la ruta `data/{asset_class}/{symbol}/{timeframe}.parquet` (por ejemplo `data/crypto/BTCUSDT/1h.parquet`). Cada escritura fusiona los datos nuevos con los ya existentes en el archivo (lee → combina → reordena → reescribe), resolviendo duplicados por timestamp a favor del dato más reciente. Crea las carpetas automáticamente si no existen. Usa `pandas` + `pyarrow` internamente, pero nunca expone un `DataFrame` fuera de la clase — solo entran y salen `MarketBar`.
- **`data_manager.py`** — `DataManager`: el orquestador. Recibe un `BarRepository` por inyección de dependencia (así no le importa si el storage real es memoria o Parquet). Expone:
  - `store_bars(instrument, timeframe, bars)`: deduplica por timestamp (si hay dos velas con el mismo timestamp, gana la última) y ordena antes de delegar al repository.
  - `get_bars(instrument, timeframe, start, end)`: recupera un rango de velas.
  - `latest_timestamp(instrument, timeframe)`: devuelve el timestamp de la vela más reciente guardada, o `None` si no hay nada. Esto es clave para las descargas incrementales.

**Para qué sirve:** separa "qué se guarda y cuándo" (reglas de negocio, en `DataManager`) de "cómo se guarda físicamente" (detalle de infraestructura, en `ParquetRepository`). Si mañana se cambia Parquet por otra tecnología (SQLite, DuckDB...), solo se escribe un nuevo Repository — `DataManager` no cambia ni una línea.

### 4.4. Capa de sincronización — `src/sync/`

- **`market_data_sync_service.py`** — `MarketDataSyncService`: es la **única** pieza del sistema que conoce tanto a un `BaseProvider` como a un `DataManager` (el propio `DataManager` tiene explícitamente prohibido hablar con providers). Su único trabajo es decidir **qué rango de tiempo hace falta pedir**:
  1. Consulta `latest_timestamp` en el `DataManager`.
  2. Si ya hay datos guardados, retoma la descarga desde ahí en vez de desde el principio (evita re-descargar todo el histórico cada vez).
  3. Si el rango pedido ya está cubierto, ni siquiera llama al provider.
  4. Si hay datos nuevos, los pide al provider y los guarda vía `DataManager.store_bars`.

**Para qué sirve:** cierra el círculo `Provider → MarketBar → DataManager → Parquet`. Es la pieza que, en la práctica, "rellena" `data/` con información real de mercado.

### 4.5. Capa de features — `src/features/`

Convierte velas OHLCV crudas (las que persiste `DataManager`) en variables numéricas listas para un modelo de ML. **Nunca** habla con providers, repositorios ni sabe para qué modelo se van a usar las columnas que produce.

- **`base_feature_calculator.py`** — `FeatureCalculator`: contrato abstracto. Define `compute(bars) -> DataFrame`: recibe una secuencia de `MarketBar` en cualquier orden y devuelve un DataFrame indexado por `timestamp` (UTC, cronológico) con las columnas OHLCV originales más una columna por feature calculada.
- **`indicator_config.py`** — `IndicatorConfig`: dataclass inmutable con todas las ventanas y parámetros (periodos de SMA/EMA, RSI, MACD, Bollinger, ATR, Stochastic, ADX, MFI, retornos, volumen relativo). Centraliza la configuración para no tener números mágicos repartidos por el código.
- **`technical_indicators.py`** — `TechnicalIndicatorCalculator`: implementación concreta de `FeatureCalculator` sobre `pandas-ta-classic`. Calcula, por categoría:
  - **Tendencia:** SMA, EMA (expresadas como `sma_N_dist`/`ema_N_dist`, la distancia relativa del precio a la media, no el valor absoluto — ver nota más abajo), ADX + DI+/DI-.
  - **Momentum:** RSI, MACD, Stochastic Oscillator, MFI.
  - **Volatilidad:** Bollinger Bands (expresadas como `bb_percent_b` y `bb_bandwidth`, no como bandas en crudo — misma razón que SMA/EMA), ATR.
  - **Volumen:** OBV, volumen relativo (volumen actual vs. su media móvil).
  - **Retornos pasados:** log-retorno sobre varias ventanas configurables (`return_1`, `return_3`...).
  - **Temporales:** hora del día y día de la semana codificados cíclicamente con seno/coseno (`hour_sin`/`hour_cos`, `dow_sin`/`dow_cos`, a partir del `timestamp` UTC) y sesión de mercado one-hot (`session_asia`, `session_europe`, `session_us`, `session_off`), con los límites horarios de cada sesión configurables en `IndicatorConfig`. Capturan patrones ligados a la sesión horaria y al día de la semana, relevantes incluso en un mercado 24/7 como cripto por diferencias de volumen/volatilidad entre sesiones.

  Las filas anteriores a que se llene la ventana de lookback de un indicador quedan en `NaN` para esa columna — es el comportamiento esperado, no un error, y lo resuelve la siguiente capa (`datasets/`) al construir la tabla de entrenamiento.

  **Por qué SMA/EMA/Bollinger van en relativo y no en crudo:** una inspección del dataset real (`X`/`y` sobre BTC/USDT) mostró que `sma_20` y `bb_mid` correlacionaban 1.000 entre sí — son literalmente el mismo número (la media móvil de 20 periodos), y todo el bloque SMA/EMA/Bollinger correlacionaba por encima de 0.99. La causa es que, aunque ya excluimos el OHLCV crudo de `X` (ver [4.6](#46-capa-de-datasets--srcdatasets)), estos tres indicadores seguían siendo versiones suavizadas del precio absoluto en dólares — el mismo problema de escala entre instrumentos que justificó excluir el precio crudo, solo que disfrazado dentro de un indicador. La corrección: expresarlos como distancia relativa al precio (`(close - sma) / sma`) y como los derivados estándar de Bollinger (`%B` y ancho de banda), que son adimensionales y sí generalizan entre instrumentos con escalas de precio distintas. MACD y ATR no se tocaron: el mismo análisis no mostró evidencia de redundancia ahí, y "arreglarlos" sin datos que lo justifiquen sería sobre-diseñar — se revisarán si en el futuro aparece evidencia similar.

**Para qué sirve:** las velas crudas no sirven directamente para entrenar un modelo. Esta capa es la que produce las variables predictivas (features), manteniendo pandas y `pandas-ta-classic` como detalle de implementación confinado a esta clase — el resto del sistema solo ve `MarketBar` a la entrada y un DataFrame a la salida.

### 4.6. Capa de datasets — `src/datasets/`

Combina las features ya calculadas con el label (variable objetivo) para producir la tabla final que consume un modelo. A diferencia de `providers/` o `database/`, no es contrato abstracto + implementación: es un orquestador concreto (mismo patrón que `DataManager` o `MarketDataSyncService`), porque el proyecto solo necesita un tipo de label (retorno futuro) — no hay todavía una segunda implementación intercambiable que justifique una interfaz.

- **`dataset_builder.py`** — `DatasetBuilder`: recibe un `FeatureCalculator` por inyección de dependencia. Su método `build(bars, timeframe, horizon)`:
  1. Calcula las features con el `FeatureCalculator` inyectado.
  2. Añade la columna `label`: el **log-retorno futuro** sobre el horizonte pedido, `ln(close[t + horizon] / close[t])`. Se eligió log-retorno (no retorno porcentual simple) por coherencia con las features `return_1`/`return_3`... que ya son log-retornos, por su propiedad aditiva entre horizontes y por tratar simétricamente subidas y bajadas.
  3. Elimina filas incompletas: las iniciales sin ventana de lookback llena y las finales sin vela futura para calcular el label (recorte, no imputación).
  4. Devuelve `(X, y)` ya separados: `X` excluye a propósito las columnas OHLCV crudas (precio y volumen absolutos) porque no son comparables entre instrumentos con escalas distintas (BTC ~60.000 vs. EUR/USD ~1.08) ni estacionarias en el tiempo — solo entran columnas relativas/normalizadas (indicadores, retornos, features temporales). `y` es la serie de labels alineada por índice con `X`.
- El horizonte se expresa como `timedelta` (ej. `timedelta(hours=4)`), no como número de velas: `DatasetBuilder` lo convierte usando la nueva propiedad `TimeFrame.duration` (añadida en `src/domain/timeframe.py`), y valida que sea múltiplo exacto de la duración de una vela — pedir 30 minutos sobre velas de 1h falla explícitamente en vez de redondear en silencio. Así el horizonte es configurable sin tocar código y funciona igual para cualquier timeframe.

**Para qué sirve:** cierra el pipeline `MarketBar → Features → (X, y)`. Es la pieza que entrega la tabla de entrenamiento; la siguiente capa (`training/`) ya no necesita saber nada de indicadores, velas ni horizontes — solo consume `X`/`y`.

### 4.7. Tests — `tests/`

Cada módulo de negocio tiene su suite de tests con `unittest` (librería estándar de Python, sin dependencias nuevas):

- `tests/database/test_data_manager.py` — 7 tests: guardado/lectura, deduplicación, validaciones, aislamiento entre timeframes.
- `tests/database/test_parquet_repository.py` — 10 tests: persistencia real en disco (usando carpetas temporales, nunca `data/` real), fusión entre escrituras sucesivas, normalización de símbolos, creación automática de carpetas, y una prueba de integración con `DataManager`.
- `tests/sync/test_market_data_sync_service.py` — 6 tests: descarga inicial completa, reanudación incremental, fusión con lo ya guardado, skip cuando ya está al día, y validaciones. Usa un `FakeProvider` (un doble de test que cumple el contrato `BaseProvider`) para no depender de red real.
- `tests/features/test_technical_indicators.py` — 11 tests: presencia de todas las columnas esperadas (incluidas las temporales), orden cronológico e índice UTC independientemente del orden de entrada, `NaN` cuando no hay bars suficientes para el lookback, verificación de valores calculados a mano para SMA, log-retorno y volumen relativo, codificación cíclica de hora/día-de-semana, y flags de sesión de mercado one-hot.
- `tests/domain/test_timeframe.py` — 1 test: `TimeFrame.duration` devuelve la duración correcta para cada intervalo declarado.
- `tests/datasets/test_dataset_builder.py` — 7 tests: validaciones de horizonte (positivo, múltiplo exacto de la duración de vela, bars no vacío), exclusión de columnas OHLCV/label crudas en `X`, `X`/`y` comparten índice y no tienen `NaN`, recorte de las filas finales sin label futuro, y verificación del label calculado a mano.
- `tests/training/test_time_series_split.py` — 11 tests: validaciones del constructor (`n_splits`, `embargo`, `test_size`), `test_split` reserva la fracción final y purga el embargo, `split` genera exactamente `n_splits` folds con ventana de entrenamiento creciente, nunca mezcla futuro antes que pasado, respeta el hueco de embargo, y lanza error cuando no hay filas suficientes.
- `tests/training/test_metrics.py` — 6 tests: `directional_accuracy` con acierto total, fallo total, acierto parcial, predicción cero (nunca cuenta como acierto), y validaciones de índice/vacío.
- `tests/training/test_random_forest_trainer.py` — 7 tests: `model_id`, error si se predice o se piden `feature_importances` antes de `fit`, validaciones de `fit` (vacío, índice no alineado), predicciones indexadas igual que la entrada, e importancias ordenadas de mayor a menor.
- `tests/training/test_walk_forward_evaluator.py` — 6 tests: usa un `_FakeTrainer` (doble de test, mismo patrón que `FakeProvider`) para verificar que se pide un modelo nuevo por fold, que los tamaños de fold coinciden con el splitter, y que las métricas coinciden con lo calculado a mano; más una prueba de integración con `RandomForestTrainer` real sobre una señal sintética aprendible.
- `tests/training/test_xgboost_trainer.py` — 7 tests: mismas validaciones que `RandomForestTrainer` (id, errores antes de `fit`, índice alineado, importancias ordenadas), sobre `XGBoostTrainer`.
- `tests/evaluation/test_model_registry.py` — 2 tests: `hyperparameter_combinations` cubre el grid completo sin duplicados por modelo, y las factories producen el `model_id` esperado.
- `tests/evaluation/test_experiment_runner.py` — 5 tests: un resultado por configuración, orden de mejor a peor por accuracy direccional media (desempate por desviación), error si falta el timeframe pedido en `bars_by_timeframe`, `evaluate_holdout` exige una corrida previa, y devuelve modelo entrenado + métricas de holdout coherentes.

Total: **86 tests**, todos en verde. Se ejecutan con:

```bash
python -m unittest discover -s tests -v
```

### 4.8. Dependencias — `requirements.txt`

- `requests` — cliente HTTP usado por `BinanceProvider`.
- `pandas` y `pyarrow` — manejo tabular y lectura/escritura de Parquet, usados únicamente dentro de `ParquetRepository`.
- `pandas-ta-classic` — cálculo de indicadores técnicos, usado únicamente dentro de `TechnicalIndicatorCalculator`.
- `numpy` — cálculo de log-retornos, usado únicamente dentro de `TechnicalIndicatorCalculator`.
- `scikit-learn` — `RandomForestTrainer`, métricas de error (MAE, RMSE) y `ParameterGrid` para los barridos de hiperparámetros, usado dentro de `src/training/` y `src/evaluation/`.
- `joblib` — persistencia del modelo entrenado a disco (`scripts/train_model.py`, `scripts/run_experiments.py`). Dependencia transitiva de scikit-learn, listada explícitamente porque se importa directamente.
- `xgboost` — `XGBoostTrainer`, segunda implementación de `ModelTrainer` sobre gradient boosting.

### 4.9. Capa de training — `src/training/`

Entrena y valida modelos de ML sobre la tabla `(X, y)` que entrega `DatasetBuilder`. Es la primera capa de la arquitectura con **dos** patrones de contrato abstracto + implementación conviviendo en el mismo módulo: uno para el modelo en sí (`ModelTrainer`), y otro, más simple, de orquestador concreto para la validación (`WalkForwardEvaluator`), mismo patrón que `DatasetBuilder` o `MarketDataSyncService`.

- **`base_trainer.py`** — `ModelTrainer`: contrato abstracto. Define `model_id` (identificador estable, ej. `"random_forest"`), `fit(X, y)` y `predict(X) -> pd.Series`. Cualquier algoritmo futuro (XGBoost, LightGBM, LSTM...) solo necesita cumplir este contrato para ser intercambiable, sin tocar la validación ni los scripts que lo usan.
- **`random_forest_trainer.py`** — `RandomForestTrainer`: primera implementación concreta, sobre `sklearn.ensemble.RandomForestRegressor`. Elegido como baseline sobre XGBoost/LightGBM o redes recurrentes/Transformers por varias razones: no necesita escalar features, tolera bien la multicolinealidad residual que aún queda entre indicadores, y expone `feature_importances_` — clave para verificar rápidamente si los indicadores aportan señal real antes de invertir en arquitecturas más complejas y más difíciles de depurar. Expone además una propiedad `feature_importances` (Serie ordenada de mayor a menor) para esa inspección. En la práctica resultó ser el más lento de los dos modelos con diferencia (sklearn no usa histogramas como XGBoost) — varios minutos por fit en el dataset de 15 minutos (~312.000 filas) frente a segundos de XGBoost para un ajuste comparable.
- **`xgboost_trainer.py`** — `XGBoostTrainer`: segunda implementación concreta, sobre `xgboost.XGBRegressor` (gradient boosting: árboles secuenciales que corrigen el error de los anteriores, en vez de árboles independientes promediados como Random Forest). Mismo contrato, mismos métodos (`model_id`, `fit`, `predict`, `feature_importances`) — intercambiable sin tocar el resto del pipeline. En la búsqueda comparativa (ver [4.10](#410-capa-de-evaluation--srcevaluation)) ganó sistemáticamente a Random Forest, tanto en accuracy direccional como en velocidad de entrenamiento.
- **`time_series_split.py`** — `PurgedWalkForwardSplit`: divisor cronológico **estricto**, sin mezcla aleatoria. Dos garantías independientes:
  - `test_split(index)`: reserva el tramo final (`test_size`, ej. 15%) como **holdout** definitivo, purgado con un `embargo`. Este tramo solo se evalúa una vez, al final de todo.
  - `split(index)`: genera `n_splits` folds *walk-forward* de ventana expansiva (entrena con `[0:k]`, valida justo después de un hueco de `embargo`, repite avanzando `k`), sobre el resto de los datos ("dev").

  El **embargo** existe porque el label de `DatasetBuilder` mira hacia el futuro (`horizon` velas por delante): sin un hueco purgado en cada frontera train/validación, las últimas filas de entrenamiento podrían tener un label calculado con un precio que cae dentro de la ventana de validación — fuga de información sutil que un split cronológico "ingenuo" no evita. El embargo se pasa en número de velas (igual a `horizon // timeframe.duration`).
- **`metrics.py`** — `directional_accuracy(y_true, y_pred)`: única métrica no cubierta por scikit-learn (MAE/RMSE se reutilizan directamente de `sklearn.metrics`, sin reimplementar). Mide qué fracción de las predicciones acierta el signo (sube/baja) — la métrica que de verdad importa para una señal de trading, más que el error absoluto.
- **`walk_forward_evaluator.py`** — `WalkForwardEvaluator`: orquestador concreto (sin contrato abstracto — solo hay una forma de correr la validación por ahora) inyectado con un `PurgedWalkForwardSplit` y una *factory* de `ModelTrainer`. Por cada fold pide un modelo **nuevo y sin entrenar** a la factory — reutilizar el mismo modelo entre folds reintroduciría la fuga que el splitter existe para evitar. Devuelve una lista de `FoldResult` (tamaño de train/val, MAE, RMSE, directional accuracy) por fold, para ver el rendimiento por régimen de mercado en vez de un único número agregado.

**Para qué sirve:** cierra el pipeline `(X, y) → Modelo entrenado y validado`. La capa de evaluación (ver [4.10](#410-capa-de-evaluation--srcevaluation)) construye reportes y comparaciones formales entre modelos sobre esta base; esta capa ya deja la validación *estricta* resuelta (nunca aleatoria, con embargo, con holdout final intacto), no solo el entrenamiento.

**Primer experimento — BTC/USDT 1h, horizonte 4h:** `scripts/train_model.py` corre el pipeline completo sobre el histórico completo de BTC/USDT (77.777 filas limpias). Split: 15% final reservado como test (2025-03-16 → 2026-07-15, nunca visto durante la validación), embargo de 4 velas (= horizonte), 5 folds walk-forward sobre el 85% restante ("dev", 2017-08-25 → 2025-03-16).

**Resultado:** *directional accuracy* del **48.4%–50.3%** en los 5 folds (media 49.2%, std 0.7%) y **48.4%** en el test final — es decir, **no mejor que lanzar una moneda al aire**, y de hecho ligeramente por debajo en la mayoría de los folds. MAE/RMSE (0.006–0.019 en log-retorno) están en línea con la volatilidad típica de BTC/USDT a 4h, no son anormales. Las features más importantes según el modelo (`bb_bandwidth`, `return_10`, `sma_200_dist`, `return_5`, `ema_12_dist`...) son razonables, pero no se traducen en poder predictivo direccional.

**Qué significa esto:** no es un bug ni un error de implementación — es exactamente lo que la validación estricta está diseñada para revelar de forma honesta. Confirma, con un modelo real y validación rigurosa, la señal débil que ya había aparecido en la EDA inicial sobre correlaciones feature→target (ver también la nota de multicolinealidad en [4.5](#45-capa-de-features--srcfeatures)): los indicadores técnicos clásicos por sí solos, sobre BTC/USDT a 4h, no bastan para predecir la dirección mejor que el azar con un Random Forest. Es un resultado de baseline legítimo, no el final del camino.

**Qué queda fuera deliberadamente (no bloqueante para seguir):** un solo horizonte por `ModelTrainer` a la vez — probar varios horizontes simultáneamente en una sola llamada no se soportó porque `DatasetBuilder.build()` ya obliga a un horizonte por invocación (ver [4.6](#46-capa-de-datasets--srcdatasets)); el barrido de horizontes se resuelve en la capa de evaluación ([4.10](#410-capa-de-evaluation--srcevaluation)) llamando a `build()` una vez por combinación.

### 4.10. Capa de evaluation — `src/evaluation/`

Compara sistemáticamente combinaciones de modelo, hiperparámetros, timeframe y horizonte, sin comprometer la validación estricta de `src/training/`: toda la búsqueda ocurre **solo** contra los folds walk-forward del dev set; el holdout de cada dataset se evalúa una única vez, para la configuración ganadora, al final de todo. Probar muchas configuraciones y quedarse con la que mejor puntúe contra el mismo test set reintroduciría en silencio el mismo sesgo de *data snooping* que el split purgado existe para evitar — separar "buscar" de "confirmar" es lo que mantiene la búsqueda amplia honesta.

- **`experiment.py`** — `ExperimentConfig` (un punto del grid: `model_name`, `timeframe`, `horizon`, `hyperparams`) y `ExperimentResult` (media/desviación de directional accuracy y de MAE/RMSE a través de los folds de dev para ese punto). Nunca contienen información del holdout.
- **`model_registry.py`** — `TRAINER_FACTORIES` (nombre de modelo → función que construye un `ModelTrainer`) y `HYPERPARAMETER_GRIDS` (qué combinaciones probar por modelo). Los grids se mantuvieron pequeños a propósito (4 combinaciones por modelo: 2 valores de nº de árboles × 2 de profundidad) — el objetivo era ver si cambiar la profundidad o el número de árboles alteraba el panorama, no exprimir una décima de mejora con una búsqueda densa.
- **`experiment_runner.py`** — `ExperimentRunner`: orquestador concreto (mismo patrón que `DatasetBuilder`/`WalkForwardEvaluator` — sin contrato abstracto, solo hay una forma de correr la búsqueda). Construye y cachea un `Dataset` (X/y de dev y de holdout, más el splitter) por cada combinación distinta de `(timeframe, horizon)`, para no recalcular features de más entre hiperparámetros que comparten dataset. `run(configs, bars_by_timeframe)` evalúa cada configuración contra sus folds de dev y devuelve los resultados ordenados de mejor a peor (por accuracy direccional media, desempatando por menor desviación entre folds — se prefiere un resultado algo peor pero más estable a uno mejor pero errático). `evaluate_holdout(config)` solo se llama una vez, sobre la configuración ganadora ya elegida, y devuelve el modelo entrenado con todo el dev y sus métricas de holdout.

**Para qué sirve:** cierra la pregunta "¿qué configuración es mejor, y de verdad, no por suerte de la búsqueda?" — sin esta capa, comparar 32 configuraciones a mano una por una (como se hizo para el primer baseline) no escala y tienta a mirar el holdout más de una vez.

**Búsqueda completa — BTC/USDT, 2 modelos × 4 hiperparámetros × 4 combinaciones timeframe/horizonte:** `scripts/run_experiments.py` corrió las 32 configuraciones sobre el histórico completo (`15m→30min`, `1h→4h`, `4h→24h`, `1d→3 días`), 5 folds walk-forward cada una, guardando el ranking completo en `reports/experiment_results.csv`.

**Resultado:**

| Timeframe→Horizonte | Mejor accuracy direccional (dev) | Notas |
|---|---|---|
| 15m → 30min | **51.6%** (XGBoost, depth=3, n=200) | Mejor combinación con diferencia; std bajo (0.85%) entre folds |
| 1h → 4h | 51.0% (XGBoost) | Random Forest se quedó en ~49% en este mismo horizonte |
| 4h → 24h | 49.0% (XGBoost) | Random Forest cayó a 47.6-48.0% — peor que el azar |
| 1d → 3 días | 51.2% (Random Forest) | Solo 3.257 filas; std hasta 2.7% entre folds — poco fiable |

Ganador: **XGBoost, timeframe 15 minutos, horizonte 30 minutos, `max_depth=3, n_estimators=200`**. Holdout (evaluado una sola vez): **directional accuracy 51.8%**, mae=0.00212, rmse=0.00325 — consistente con el 51.6% visto en dev, señal de que la mejora no es pura casualidad de la búsqueda. Modelo persistido en `models/best_xgboost_15m_03000.joblib`.

**Qué significa esto:** XGBoost bate sistemáticamente a Random Forest, y los horizontes cortos (15m→30min, 1h→4h) muestran más señal que los largos (4h→24h fue peor que el azar). La mejora sobre el 50% (~1.5-2 puntos porcentuales) es consistente entre dev y holdout, pero sigue siendo pequeña — con 32 configuraciones probadas, parte de esa mejora podría deberse a suerte de la búsqueda (*multiple comparisons*); el holdout ayuda a descartar que sea *solo* eso, pero no lo garantiza al 100%. No es una señal lo bastante fuerte para operar con ella sin más — es un indicio real, no una estrategia.

**Qué queda fuera deliberadamente (no bloqueante para seguir):** grids de hiperparámetros pequeños (4 por modelo) — una búsqueda más densa (learning rate, subsample, regularización...) se pospone porque con una señal tan débil, afinar hiperparámetros probablemente mida ruido con más precisión antes que crear señal donde no la hay. Tampoco se corrigió la significancia estadística de forma rigurosa (ej. bootstrap sobre los folds, corrección por comparaciones múltiples) — el holdout de un solo uso es la salvaguarda actual, no una prueba formal.

**Modelos guardados por combinación:** `scripts/run_experiments.py` ahora persiste el ganador de **cada** combinación timeframe/horizonte (no solo el ganador global), en `models/best_{modelo}_{timeframe}_{horizonte}.joblib`. Al re-evaluar los 4 contra su propio holdout se detectó algo importante: el de `1d→3 días`, que en dev parecía competitivo (51.2%), **cae a 47.6% en su holdout** — por debajo del azar. Confirma que ese resultado era ruido de un dataset pequeño (3.257 filas), no señal real; los otros tres (15m, 1h, 4h) se mantienen razonablemente estables entre dev y holdout.

**Backtest con costes reales — `backtest.py`:** antes de construir `src/prediction/`, se añadió `directional_backtest(y_true, y_pred, cost_per_trade)`: simula una regla de trading fija (posición larga/corta según el signo de la predicción, una operación de ida y vuelta por fila, sin posiciones solapadas — un backtest vectorizado de primera pasada, no un simulador de cartera) y resta un coste por operación. La regla y el coste se fijaron **antes** de mirar el resultado, precisamente para no repetir el problema de *data snooping* que la búsqueda de hiperparámetros ya evitó — probar varios umbrales o costes hasta encontrar uno favorable reintroducría el mismo sesgo. 7 tests en `tests/evaluation/test_backtest.py`.

**Resultado (`scripts/backtest_costs.py`, comisión estándar de Binance Spot: 0.1% por ejecución × 2 = 0.2% ida y vuelta, sin descuento por BNB ni volumen):**

| Timeframe→Horizonte | Nº operaciones | Bruto | Neto | Win rate neto |
|---|---|---|---|---|
| 15m→30min | 46.779 | +407.8% | **-8.948%** | 19.2% |
| 1h→4h | 11.666 | -31.7% | -236.5% | 35.7% |
| 4h→24h | 2.897 | -48.4% | -627.8% | 44.8% |
| 1d→3 días | 458 | -164.4% | -256.0% | 43.9% |

**Qué significa esto:** los cuatro pierden dinero neto sobre su holdout. El caso más claro es 15m→30min: la ventaja media por operación era de **0.87 puntos básicos** en bruto, frente a un coste de **20 puntos básicos** por operación — el coste es ~23 veces mayor que el edge. Operar en cada vela de 15 minutos con un edge de ese tamaño nunca puede superar comisiones estándar, sin importar cuánto se afine el modelo. Esto confirma, con datos, que **no conviene construir `src/prediction/` sobre estos modelos tal cual** — sería servir una estrategia que pierde dinero por diseño, no por error de implementación.

**Qué queda fuera deliberadamente (no bloqueante para seguir):** filtrar operaciones por confianza (solo operar cuando `|predicción|` supera un umbral, para reducir frecuencia y quedarse con las señales más fuertes) no se probó aquí — hacerlo sobre el mismo holdout ya usado reintroduciría exactamente el sesgo que se ha evitado todo este tiempo. Si se persigue esa idea, el umbral debe elegirse con los folds de dev (nunca con el holdout) y confirmarse una sola vez, igual que el resto de decisiones de este documento.

**Filtrado por confianza — resultado (`scripts/threshold_search.py`):** se probó justo esa idea sobre el ganador 15m→30min: elegir, solo con los folds de dev, qué fracción de las predicciones de mayor magnitud conviene operar (`1.0, 0.5, 0.25, 0.1, 0.05`). Ni siquiera la fracción más agresiva (`0.05`, operar solo el 5% de mayor confianza) logró retorno neto positivo en dev. Confirmado una sola vez sobre el holdout con el umbral elegido: sigue en pérdidas. El edge del modelo es demasiado débil y demasiado disperso entre operaciones para que concentrarse en las "mejores" señales lo vuelva rentable — no es un problema de frecuencia de operación, es que el propio modelo no separa bien ganadoras de perdedoras ni siquiera en su cola de mayor confianza.

### 4.11. Contexto multi-timeframe — intento y resultado

**Qué es:** `MultiTimeframeFeatureCalculator` (`src/features/multi_timeframe_feature_calculator.py`), la palanca identificada como "próximo paso" tras ver que ni el modelo base ni el filtrado por confianza bastaban: añadir al dataset de 15m un RSI y una distancia a SMA larga calculados sobre velas de 4h y de 1d, como contexto de tendencia macro (ver [6.2](#62-srcdatasets--dataset-builder--hecho)). Envuelve un `FeatureCalculator` base (el de 15m) y, por cada timeframe de contexto configurado, calcula sus propios indicadores y los fusiona con `pd.merge_asof`, desplazados a su hora de cierre — una vela de 4h solo es conocible una vez cerrada, nunca en su apertura, para no filtrar información del futuro. 5 tests en `tests/features/test_multi_timeframe_feature_calculator.py`.

**Búsqueda (`scripts/multi_timeframe_experiment.py`):** mismo modelo ganador (XGBoost, `max_depth=3, n_estimators=200`) y mismo par timeframe/horizonte (15m→30min), pero con el dataset ampliado a 36 columnas (las de siempre más `rsi_14_4h`, `sma_200_dist_4h`, `rsi_14_1d`, `sma_200_dist_1d`). Siguiendo la misma disciplina del resto del documento, el holdout solo se toca si el resultado en dev supera al baseline de una sola señal.

**Resultado:**

| | Dev (5 folds walk-forward) | Holdout |
|---|---|---|
| Solo 15m (baseline, [4.10](#410-capa-de-evaluation--srcevaluation)) | 51.6% | 51.8% |
| 15m + contexto 4h/1d | **51.77%** (std 0.19%) | **51.88%** |

La mejora (~0.15-0.2 puntos porcentuales) es real y consistente entre dev y holdout — no es ruido de una sola tirada — pero es demasiado pequeña para importar en la práctica. El filtrado por confianza repetido sobre este dataset ampliado (misma búsqueda de `keep_fraction` que en [4.10](#410-capa-de-evaluation--srcevaluation)) tampoco encuentra ningún umbral con retorno neto positivo en dev; incluso en el 5% de mayor confianza el retorno neto medio por operación sigue en -17.2 puntos básicos, frente a un coste de 20. En el holdout, con ese mismo umbral: 1.311 operaciones, retorno neto -18.1 puntos básicos por operación, -237% acumulado, win rate 33.3% (mejor que el 18.6% sin filtrar, pero lejos del ~55%+ que haría falta para cubrir el coste con este tamaño de edge).

**Qué significa esto:** añadir contexto de timeframes superiores ayuda un poco — confirma que hay algo de información real en la tendencia macro que el modelo de 15m no veía — pero no cambia la conclusión de fondo. El problema no es que falten features de un tipo concreto (multi-timeframe, en este caso); es que los indicadores técnicos clásicos, solos o combinados entre escalas temporales, no producen un edge direccional lo bastante grande para un instrumento tan líquido y con comisiones estándar de Binance a granularidad de 15-30 minutos. Seguir iterando sobre este mismo tipo de features (más indicadores, más timeframes de contexto) tiene rendimientos decrecientes esperables, no una promesa de cruzar el umbral de rentabilidad.

**Qué queda fuera deliberadamente (no bloqueante para seguir):** no se repitió esta búsqueda sobre los pares 1h→4h ni 4h→24h — el de 15m→30min ya tenía el edge más prometedor de los cuatro en el baseline, y el patrón de resultado (mejora pequeña, filtrado por confianza sigue sin ser rentable) no da motivo para esperar algo distinto en los demás. Tampoco se probaron más columnas de contexto por timeframe (solo RSI + tendencia larga, ver la clase) — añadir más aumentaría la dimensionalidad sin que haya evidencia de que el cuello de botella sea "pocas features" en vez de "las features clásicas no alcanzan".

### 4.12. Funding rate de futuros perpetuos — intento y resultado

**Qué es:** tras descartar tanto el filtrado por confianza como el contexto multi-timeframe, y comprobar que bajar comisión tampoco cierra la brecha ([9](#9-próximo-paso-inmediato)), se probó una fuente de información cualitativamente distinta: el *funding rate* de BTCUSDT perpetuo en Binance Futures — lo que pagan los largos a los cortos (o al revés) cada 8h, una señal real de presión de posicionamiento documentada en trading cuant, que no se deriva del precio spot como todos los indicadores usados hasta ahora.

**Qué se implementó (piezas nuevas, mismo patrón contrato+implementación del resto del proyecto):** `FundingRate` (entidad de dominio) · `FundingRateProvider`/`BinanceFundingRateProvider` (descarga desde `fapi.binance.com/fapi/v1/fundingRate`) · `FundingRateRepository`/`ParquetFundingRateRepository` (persistencia en `data/crypto/BTCUSDT/funding_rate.parquet`, mismo árbol que los timeframes) · `FundingRateFeatureCalculator` (añade `funding_rate` y `funding_rate_cum_3` — suma de las últimas 3 liquidaciones, 24h acumuladas — fusionadas por `merge_asof`; a diferencia del contexto multi-timeframe, aquí no hace falta desplazar a "hora de cierre": el momento de liquidación ya es el instante en que el dato se conoce). 25 tests nuevos entre dominio, provider (con sesión HTTP falsa), repositorio y feature calculator. Backfill real ejecutado: 7.516 liquidaciones desde el lanzamiento del perpetuo (2019-09-10) hasta hoy.

**Búsqueda (`scripts/funding_rate_experiment.py`):** mismo modelo ganador y mismo par 15m→30min, mismo gate que en 4.11 (no toca holdout si dev no supera 51.6%).

**Resultado:** el gate **no se superó** — accuracy direccional en dev del **51.36%** (std 0.93%, notablemente más ruidosa entre folds que el 0.19% visto con contexto multi-timeframe), por debajo del baseline de una sola señal. El script paró ahí, sin tocar el holdout, tal como está diseñado.

**Qué significa esto:** a diferencia del contexto multi-timeframe (que sí mejoraba dev aunque no bastara para ser rentable), el funding rate ni siquiera superó el baseline en este primer intento. Un matiz honesto a tener en cuenta antes de descartar la idea del todo: el histórico de funding rate solo llega hasta 2019-09-10 (lanzamiento del perpetuo), frente a los datos de precio que llegan hasta 2017 — el dataset resultante tiene 239.992 filas en vez de las ~312.000 habituales, un periodo más corto y con menos variedad de regímenes de mercado, lo que por sí solo podría explicar parte del peor resultado y de la mayor dispersión entre folds, independientemente de si el funding rate en sí aporta o no señal.

**Qué queda fuera deliberadamente (no bloqueante para seguir):** no se repitió el experimento restringiendo el resto de features al mismo rango de fechas 2019-2026 para aislar si la caída se debe al feature o al periodo más corto — sería el siguiente paso natural si se quisiera insistir en esta vía en vez de pasar a la número 3 (replantear el objetivo).

### 4.13. Predecir volatilidad en vez de dirección — Fase 1: ¿es predecible? (primer resultado positivo del proyecto)

**Qué es:** tras agotar dos vías para mejorar la predicción de *dirección* (4.11, 4.12), se cambió de objetivo: en vez de predecir si el precio sube o baja, predecir **cuánto se va a mover** (volatilidad realizada futura, `sqrt(suma de retornos log al cuadrado sobre el horizonte)`), una magnitud siempre positiva que la literatura cuant respalda como más predecible que la dirección (clustering de volatilidad, la base de modelos GARCH). Antes de construir infraestructura de opciones (que necesitaría datos de Deribit — DVOL, su índice de volatilidad implícita a 30 días, gratuito y verificado como viable), esta Fase 1 comprobó barato si hay algo real que predecir, comparando un XGBoost contra una baseline ingenua de persistencia ("la volatilidad de la ventana pasada predice la de la ventana futura").

**Qué se implementó:** `DatasetBuilder` ahora acepta un `label_fn` inyectable (antes tenía una fórmula fija — se añadió el punto de extensión justo cuando apareció un segundo tipo de label real que lo justificaba, no antes) y `realized_volatility_label` (`src/datasets/labels.py`). El experimento (`scripts/realized_volatility_experiment.py`) corre su propio bucle walk-forward (no reutiliza `WalkForwardEvaluator`, acoplado a accuracy direccional, que no aplica a un target siempre positivo), midiendo R² y correlación de Pearson. 5 tests nuevos.

**Resultado — 15m→30min (312.063 velas, el mismo horizonte usado en todo el proyecto):**

| | Dev (media 5 folds) | Holdout |
|---|---|---|
| Modelo (XGBoost) | R²=0.108 | **R²=0.195, corr=0.52** |
| Baseline de persistencia | R²=-0.169 | R²=-0.162, corr=0.42 |

El modelo bate a la baseline con claridad y de forma consistente en los 5 folds de dev (positivo en los 5; la baseline es negativa en 4 de 5, peor que predecir la media). Un R² de 0.195 y correlación de 0.52 es, con diferencia, **el resultado más fuerte de todo el proyecto** — nada que ver con el 51-52% de accuracy direccional que rondaba el azar.

**Resultado — 1d→30 días (3.028 filas, el horizonte que haría falta para operar contra DVOL):**

| | Dev (media 3 folds) | Holdout |
|---|---|---|
| Modelo (XGBoost) | R²=-0.584 | R²=-0.429, corr=0.21 |
| Baseline de persistencia | R²=-0.836 | R²=-0.869, corr=0.13 |

Aquí el modelo bate a la baseline (que es aún peor), pero **ambos tienen R² negativo** — peor que predecir simplemente la media histórica. Hay algo de correlación (0.21 en holdout), pero no un ajuste fiable. Con solo 3.028 filas y 3 folds, el resultado es además mucho más ruidoso que el de 15m.

**Qué significa esto (y por qué no es un "sí, seguimos" simple):** la volatilidad **sí es predecible en este dataset** — la premisa central de la Fase 1 queda confirmada, y con el resultado más contundente visto en el proyecto. Pero hay un desajuste real: el horizonte donde hay señal fuerte (30 minutos) no es el horizonte al que cotiza DVOL (30 días, la única fuente de "precio de mercado de la volatilidad" gratuita y verificada), y el horizonte que sí encajaría con DVOL no muestra señal fiable con los datos disponibles. Operar la señal de 30 minutos vía opciones exigiría opciones de vencimiento muy corto (Deribit las tiene, pero con muchísima menos liquidez e historial gratuito verificado que DVOL a 30 días) — no es la ruta barata que se investigó.

**Qué queda fuera deliberadamente (decisión pendiente, no bloqueante):** no se ha construido nada de Deribit/DVOL ni motor de backtest de opciones todavía. La Fase 2 tal como se planteó originalmente (comparar predicción a 30 días contra DVOL) no está justificada con este resultado. La alternativa más prometedora con la señal de 30 minutos que sí funciona es monetizarla sin opciones: usarla para decidir cuándo operar el modelo direccional existente (mismo patrón que el filtrado por confianza de 4.10/4.11, pero filtrando por volatilidad esperada en vez de por la magnitud de la propia predicción direccional) — pendiente de decidir con el usuario antes de construir.

**Barrido ampliado — ¿hay un horizonte intermedio (ej. semanal) que funcione mejor?** Dado que las opciones semanales son el vencimiento más líquido de Deribit (más que 30 días o vencimientos ultra-cortos), se amplió el experimento a un barrido completo de horizontes construidos sobre velas de 1h (~78.000 filas — igual de densas para cualquier horizonte, a diferencia de construir un horizonte semanal sobre velas diarias, que solo daría ~3.000 filas):

| Horizonte | Modelo R² (holdout) | Modelo corr | Baseline R² | Baseline corr |
|---|---|---|---|---|
| 15m→30min | 0.195 | 0.520 | -0.162 | 0.419 |
| 1h→4h | **0.265** | 0.552 | -0.302 | 0.349 |
| 1h→1d | **0.235** | **0.608** | -0.097 | 0.452 |
| 1h→3d | -0.039 | 0.528 | -0.127 | 0.437 |
| 1h→7d | -0.536 | 0.313 | -0.184 | 0.438 |
| 1h→14d | -0.807 | 0.059 | -0.556 | 0.349 |
| 1d→30d | -0.429 | 0.206 | -0.869 | 0.126 |

**Resultado:** la señal real está concentrada en horizontes cortos — de 30 minutos a **1 día** el modelo bate con claridad a la baseline en R² y en correlación, y el mejor resultado de todo el barrido es 1h→1d (R²=0.235, corr=0.608). A partir de 3 días la cosa se degrada rápido: en 1h→7d y 1h→14d el modelo no solo deja de batir a la baseline, **queda peor que ella** (R² más negativo). Los folds de dev en esos horizontes muestran errores puntuales enormes (R² de fold hasta -7), probablemente por cambios de régimen (crashes, colapsos) que el modelo no anticipa a esa distancia — mientras que la correlación se mantiene razonable hasta 7 días (0.31) y solo se hunde en 14 días (0.06), lo que sugiere que el modelo "acierta el orden" (más o menos vol que lo normal) incluso cuando falla en la magnitud exacta.

**Qué significa esto:** la hipótesis de operar en semanal **no se sostiene con estos datos** — 7 días ya muestra degradación clara, y 14 días prácticamente no tiene señal. El punto dulce real es más corto de lo que ofrecen los vencimientos líquidos habituales de Deribit (semanal como mínimo) — así que el desajuste de vencimiento con el mercado de opciones sigue ahí, solo que ahora sabemos que es peor de lo que parecía: ni siquiera semanal encaja con el horizonte donde el modelo funciona bien.

### 4.14. ¿Y predecir *dirección* (no volatilidad) a intervalos más largos, tipo semanal?

**Qué es:** pregunta relacionada pero distinta a 4.13 — no volatilidad, sino volver a la dirección (subida/bajada) de siempre, pero operando con menos frecuencia. Ya había un indicio en [4.10](#410-capa-de-evaluation--srcevaluation): `1d→3días` parecía competitivo en dev (51.2%) pero se hundía a 47.6% en holdout, achacado a ruido de un dataset pequeño (3.257 filas). Se repitió correctamente esta vez, con el mismo truco que en 4.13: construir los horizontes largos sobre velas de 1h (~78.000 filas) en vez de velas diarias, para tener una muestra grande y fiable — reutilizando sin cambios `WalkForwardEvaluator`, `XGBoostTrainer` y `directional_accuracy` ya existentes (`scripts/direction_horizon_sweep.py`).

**Resultado — accuracy direccional, hiperparámetros por defecto:**

| Horizonte | Dev (media 5 folds) | Holdout |
|---|---|---|
| 1h→4h | 50.58% | 50.34% |
| 1h→1d | 49.01% | 50.32% |
| 1h→3d | 48.36% | 46.59% |
| 1h→7d | 47.96% | 46.40% |
| 1h→14d | 46.88% | 46.18% |

**Resultado inequívoco: cuanto más largo el horizonte, peor la accuracy direccional — de forma monótona.** No es ruido de dataset pequeño (esta vez con ~78.000 filas): la degradación es clara y consistente en dev y holdout por igual, y a partir de 3 días la accuracy cae claramente por debajo del azar (llegando a 46.2% en 14 días — peor que lanzar una moneda de forma sistemática, no solo "sin edge"). Confirma con una muestra grande lo que el resultado de `4h→24h` en 4.10 ya apuntaba con una muestra más pequeña.

**Qué significa esto:** operar dirección a intervalos más espaciados (semanal o más) **no es viable** — es la dirección equivocada de la palanca. La dirección de BTC/USDT es más predecible cuanto más corto el horizonte (aunque siga sin ser explotable por costes, ver [9](#9-próximo-paso-inmediato)), no menos; alargar el horizonte para operar con menos frecuencia empeora la señal en vez de simplemente reducir el número de operaciones. Esto es coherente con 4.13: la señal predecible en este mercado (tanto en dirección como en volatilidad) vive en horas, no en días ni semanas.

---

## 5. Cómo encaja todo: flujo de punta a punta (ejemplo real)

Supongamos que queremos tener el histórico de `BTC/USDT` en velas de 1 hora, actualizado hasta hoy:

1. Se crea un `Instrument(symbol="BTC/USDT", asset_class=AssetClass.CRYPTO)`.
2. Se crea un `BinanceProvider()` (implementa `BaseProvider`).
3. Se crea un `ParquetRepository(base_path=Path("data"))` (implementa `BarRepository`).
4. Se crea un `DataManager(repository)` con ese repository.
5. Se crea un `MarketDataSyncService(provider, data_manager)`.
6. Se llama a `service.sync_bars(instrument, TimeFrame.ONE_HOUR, start, end)`.
   - El servicio mira si ya hay algo guardado en `data/crypto/BTCUSDT/1h.parquet`.
   - Si es la primera vez, pide todo el rango `[start, end)` a Binance.
   - Si ya había datos hasta, digamos, ayer a las 18:00, solo pide desde ahí hasta `end`.
   - El provider devuelve una lista de `MarketBar`.
   - El `DataManager` deduplica y ordena, y le pide al `ParquetRepository` que los persista.
   - El `ParquetRepository` lee el archivo existente, fusiona, y reescribe `data/crypto/BTCUSDT/1h.parquet` ordenado y sin duplicados.
7. La próxima vez que se llame a `sync_bars` con un `end` más reciente, solo se descargará lo nuevo.

Ninguna de estas piezas conoce a las demás más allá del contrato que le corresponde: el dominio no sabe de Binance ni de Parquet; `BinanceProvider` no sabe que existe un `DataManager`; `ParquetRepository` no sabe que existe Binance; `DataManager` no sabe que existe ningún provider.

---

## 6. Qué falta por construir (hoja de ruta detallada)

Todo esto sigue el pipeline objetivo del proyecto:

```
Provider → MarketBar → DataManager → Parquet Storage   ✅ HECHO
                                          ↓
                                Feature Engineering      ✅ HECHO
                                          ↓
                                  Dataset Builder         ✅ HECHO
                                          ↓
                                      Training             ✅ HECHO (RandomForest + XGBoost)
                                          ↓
                                    Evaluation              ✅ HECHO (búsqueda exhaustiva: ~51.6% dev / 51.8% holdout)
                                          ↓
                                    Prediction                ⬜
                                          ↓
                                    Application                 ⬜
```

### 6.1. `src/features/` — Feature Engineering (✅ HECHO)

**Qué es:** a partir de las velas OHLCV crudas guardadas en Parquet, calcula variables (features) que un modelo de ML pueda usar para predecir. Ver el detalle completo en [4.5](#45-capa-de-features--srcfeatures).

**Qué se implementó:** `FeatureCalculator` (contrato abstracto) + `TechnicalIndicatorCalculator` (implementación sobre `pandas-ta-classic`) + `IndicatorConfig` (parámetros centralizados). Cubre tendencia (SMA, EMA, ADX), momentum (RSI, MACD, Stochastic, MFI), volatilidad (Bollinger, ATR), volumen (OBV, volumen relativo), retornos pasados y features temporales (hora del día y día de la semana cíclicos, sesión de mercado one-hot). 11 tests en `tests/features/test_technical_indicators.py`.

**Qué queda fuera deliberadamente (no bloqueante para seguir):**

- **Manejo de los `NaN` iniciales** que dejan los indicadores mientras se llena su ventana de lookback: es responsabilidad de la siguiente capa (`datasets/`), que decide si recorta esas filas o las imputa al construir la tabla final `X`/`y`.

### 6.2. `src/datasets/` — Dataset Builder (✅ HECHO)

**Qué es:** combina las features generadas con la variable objetivo (el **label**): el log-retorno futuro sobre un horizonte configurable (ej. "cuánto sube o baja el precio en las próximas 4 horas"). Construye la tabla final `X` (features) + `y` (target) que se le da a un modelo. Ver el detalle completo en [4.6](#46-capa-de-datasets--srcdatasets).

**Qué se implementó:** `DatasetBuilder` (orquestador concreto, sin contrato abstracto — solo hay un tipo de label por ahora) inyectado con un `FeatureCalculator`. El horizonte se pasa como `timedelta` y se convierte a número de velas vía la nueva `TimeFrame.duration`, validando que sea múltiplo exacto. `X` excluye las columnas OHLCV crudas por no ser comparables entre instrumentos; `y` es el label alineado por índice. Filas sin lookback completo o sin vela futura para el label se recortan. Validado contra el histórico completo de BTC/USDT en Binance (`data/crypto/BTCUSDT/1h.parquet`, 77.980 velas desde su listing en 2017-08-17 hasta hoy — ver el backfill de histórico descrito justo debajo): produce 77.777 filas limpias, sin `NaN`, con estadísticas de `y` razonables. 7 tests en `tests/datasets/test_dataset_builder.py` + 1 test en `tests/domain/test_timeframe.py`.

**Backfill de histórico completo:** `scripts/sync_data.py` (vía `MarketDataSyncService`) solo avanza hacia adelante desde el último dato guardado — no sirve para rellenar historia anterior a la ya almacenada. Con solo 6 meses de histórico inicial, el dataset tenía poca variedad de regímenes de mercado para entrenar y validar de forma robusta. Se añadió `scripts/backfill_data.py`, que llama directamente a `BinanceProvider.get_historical_bars` + `DataManager.store_bars` (sin pasar por el servicio de sync, que está pensado para avanzar, no retroceder) para traer todo el histórico desde el listing de BTC/USDT en Binance (2017-08-17). Resultado: de 4.320 a 77.980 velas de 1h. Se detectaron 28 huecos menores (~128 horas de ~78.000, todos entre 2017-2020, downtime típico de exchange en sus primeros años) — no se corrigen porque `DataManager` v1 ya decidió deliberadamente no implementar relleno de gaps por no hacer falta, y el volumen es insignificante.

**Segundo timeframe — 15 minutos:** el objetivo del proyecto incluye explícitamente predecir sobre un horizonte de 30 minutos (ver [sección 1](#1-qué-es-market-predictor)), pero `DatasetBuilder` exige que el horizonte sea múltiplo exacto de la duración de la vela — imposible con solo velas de 1h. `scripts/sync_data.py` y `scripts/backfill_data.py` ahora iteran sobre una tupla `TIMEFRAMES` en vez de un único `TimeFrame`. Se añadió `TimeFrame.FIFTEEN_MINUTES` (divide exacto en 30 min): 312.063 velas (2017-08-17 → hoy), huecos igual de insignificantes (33 filas, ~565 slots de 15 min sobre 312.000). Validado: `DatasetBuilder.build(bars_15m, TimeFrame.FIFTEEN_MINUTES, timedelta(minutes=30))` ya no lanza error y produce 311.862 filas limpias.

**Timeframes adicionales — 4 horas y 1 día:** distinto motivo al de 15m. Los horizontes de 4h/24h ya se pueden calcular sobre velas de 1h (4 y 24 velas respectivamente) — no hacían falta velas nativas de 4h/1d para eso. La razón real para tenerlas es otra: un indicador calculado sobre velas diarias nativas (ej. RSI diario) mide tendencia macro, información distinta a la del mismo indicador sobre velas de 1h — útil para features multi-timeframe (tendencia macro + entrada a corto plazo). Esa capa de features (`MultiTimeframeFeatureCalculator`) ya se construyó y se probó — ver [4.11](#411-contexto-multi-timeframe--intento-y-resultado); se descargaron 19.524 velas de 4h y 3.257 de 1d desde el principio para no depender de red al construirla.

**Qué queda fuera deliberadamente (no bloqueante para seguir):** un solo horizonte por llamada a `build()` — si en el futuro se quiere entrenar sobre varios horizontes a la vez, se llama a `build()` una vez por horizonte; no se ha añadido soporte multi-horizonte porque no hay evidencia todavía de que haga falta.

### 6.3. `src/training/` — Entrenamiento (✅ HECHO)

**Qué es:** el módulo que entrena modelos de ML sobre los datasets construidos, con validación cronológica estricta (walk-forward purgado con embargo, holdout final intacto). Ver el detalle completo en [4.9](#49-capa-de-training--srctraining).

**Qué se implementó:** `ModelTrainer` (contrato abstracto: `model_id`, `fit`, `predict`) + `RandomForestTrainer` y `XGBoostTrainer` (dos implementaciones) + `PurgedWalkForwardSplit` (split cronológico con embargo y holdout) + `WalkForwardEvaluator` (orquestador de la validación por folds) + `directional_accuracy` (única métrica no cubierta por scikit-learn). 37 tests nuevos en `tests/training/`. Primer experimento (baseline, Random Forest, BTC/USDT 1h/horizonte 4h): *directional accuracy* ~49% (no mejor que el azar) — resultado honesto que motivó la búsqueda más amplia en [4.10](#410-capa-de-evaluation--srcevaluation).

**Qué queda fuera deliberadamente (no bloqueante para seguir):** LightGBM, CatBoost, LSTM o Transformers como terceras/cuartas implementaciones de `ModelTrainer` — se añadirían si la búsqueda de la sección 6.4 mostrara que la familia de árboles (RF/XGBoost) tiene un techo claro que otro tipo de modelo pudiera superar; de momento ambos modelos de árboles llegan a resultados similares entre sí, así que no hay evidencia de que sea ese el cuello de botella.

### 6.4. `src/evaluation/` — Evaluación (✅ HECHO)

**Qué es:** capa que compara sistemáticamente combinaciones de modelo, hiperparámetros, timeframe y horizonte contra los folds de dev, y evalúa la configuración ganadora una única vez contra su holdout. Ver el detalle completo, resultados y tabla comparativa en [4.10](#410-capa-de-evaluation--srcevaluation).

**Qué se implementó:** `ExperimentConfig`/`ExperimentResult` (un punto del grid y su resultado en dev) + `model_registry.py` (qué modelos/hiperparámetros probar) + `ExperimentRunner` (corre y cachea datasets por timeframe/horizonte, nunca toca el holdout salvo para la configuración ganadora). 7 tests nuevos en `tests/evaluation/`. Búsqueda completa corrida vía `scripts/run_experiments.py`: 32 configuraciones (2 modelos × 4 hiperparámetros × 4 combinaciones timeframe/horizonte). Ganador: XGBoost a 15m→30min, **51.6% dev / 51.8% holdout** — mejora modesta pero consistente sobre el baseline inicial (~49%) y sobre el azar (50%).

**Qué queda fuera deliberadamente (no bloqueante para seguir):** métricas específicas de estrategia (ej. Sharpe ratio de una simulación de trading con costes de transacción, calibración de la magnitud predicha más allá del signo) y corrección estadística formal por comparaciones múltiples (ej. bootstrap sobre los folds) — se añadirían si se decide perseguir esta señal como estrategia real; de momento el holdout de un solo uso es la salvaguarda contra el sobreajuste a la búsqueda, no una prueba de significancia formal.

### 6.5. `src/prediction/` — Servicio de predicción

**Qué es:** carga un modelo ya entrenado y lo usa para predecir sobre datos nuevos (no vistos en entrenamiento). Es la puerta de entrada para "úsame en producción": dado un instrumento y un horizonte, devuelve una predicción.

### 6.6. `src/config/` — Configuración

**Qué es:** todavía no existe. Necesario para centralizar parámetros como claves de API, rutas de `data/` y `models/`, horizontes de predicción por defecto, qué proveedor usar, etc., sin hardcodear valores dispersos por el código.

### 6.7. `src/utils/` — Utilidades

**Qué es:** todavía vacío. Aquí irían utilidades genéricas y reutilizables entre capas (ej. helpers de fechas) que no encajen como responsabilidad de ningún módulo de negocio concreto. Debe usarse con cuidado para no convertirse en un cajón de sastre.

### 6.8. `app/` — Aplicación

**Qué es:** la capa más externa, la que expondría el sistema a un usuario final (API REST, dashboard, CLI...). Es lo último del pipeline, depende de todo lo anterior y no al revés.

### 6.9. Providers adicionales

Con `BaseProvider` ya definido, añadir `YahooProvider` o `PolygonProvider` en el futuro es una tarea acotada: una clase nueva en `src/providers/` que traduzca la API correspondiente a `MarketBar`, sin tocar ninguna otra capa.

---

## 7. Reglas de trabajo establecidas en este proyecto

Para quien continúe el proyecto, estas son las convenciones que se han seguido hasta ahora y que conviene mantener:

1. **Explicar antes de implementar.** Antes de crear un módulo nuevo: explicar el diseño, justificar por qué sigue Clean Architecture, y mencionar alternativas consideradas.
2. **Contrato abstracto + implementación concreta.** Cada vez que el sistema necesita hablar con "el mundo exterior" (una API, un formato de archivo...), se define primero una interfaz abstracta (`ABC` de Python) y luego la implementación concreta. Esto permite sustituir piezas sin tocar el resto del sistema y facilita testear con dobles (fakes) en vez de infraestructura real.
3. **Inyección de dependencias.** Los orquestadores (`DataManager`, `MarketDataSyncService`) reciben sus colaboradores por constructor, nunca los crean ellos mismos.
4. **Los objetos de dominio son inmutables** (`@dataclass(frozen=True)`) y viajan por todas las capas — nunca se exponen estructuras de terceros (JSON crudo de una API, `DataFrame` de pandas) fuera de la capa que las genera.
5. **Docstrings estilo Google** en todas las clases y métodos públicos, con `Args`, `Returns` y `Raises` cuando aplica.
6. **Cada pieza nueva lleva sus tests**, usando dobles de test (fakes/in-memory) para no depender de red ni de disco real salvo cuando el propio test es específicamente sobre el disco (y ahí se usa un directorio temporal).
7. **No sobre-diseñar.** Se evita añadir abstracciones, validaciones o parámetros para casos hipotéticos que no se necesitan todavía (ej.: no se implementó detección de huecos/gaps en la v1 del `DataManager` porque no hacía falta aún).

---

## 8. Resumen ejecutivo (una frase por pieza)

| Pieza | Una frase |
|---|---|
| `domain/` | El vocabulario del negocio: qué es un instrumento y una vela de precio. |
| `providers/BaseProvider` | El contrato: "así se pide histórico a cualquier fuente de datos". |
| `providers/BinanceProvider` | Cumple ese contrato hablando con la API de Binance. |
| `database/BarRepository` | El contrato: "así se guarda y se lee un histórico de velas". |
| `database/InMemoryRepository` | Cumple ese contrato en memoria, para tests. |
| `database/ParquetRepository` | Cumple ese contrato en archivos Parquet reales, en disco. |
| `database/DataManager` | Decide qué guardar y cómo (dedup, orden), sin saber cómo se persiste físicamente. |
| `sync/MarketDataSyncService` | Conecta un provider con el DataManager: descarga solo lo que falta. |
| `features/TechnicalIndicatorCalculator` | Convierte velas crudas en variables predictivas: tendencia, momentum, volatilidad, volumen, retornos y temporales (hora/día/sesión). |
| `datasets/DatasetBuilder` | Combina features + log-retorno futuro (target) en la tabla `X`/`y` de entrenamiento. |
| `training/ModelTrainer` | El contrato: "así se entrena y predice con cualquier modelo intercambiable". |
| `training/RandomForestTrainer` / `XGBoostTrainer` | Cumplen ese contrato — bagging vs. gradient boosting. XGBoost gana sistemáticamente. |
| `training/PurgedWalkForwardSplit` | Split cronológico estricto: walk-forward con embargo y holdout final, sin mezclar nunca pasado y futuro. |
| `training/WalkForwardEvaluator` | Entrena y mide un modelo nuevo por fold, sin fugas entre ellos. |
| `evaluation/ExperimentRunner` | Busca sistemáticamente modelo × hiperparámetros × timeframe/horizonte, solo contra dev; el holdout se mira una vez. |
| `prediction/` (pendiente) | Sirve predicciones con un modelo ya entrenado. |
| `app/` (pendiente) | Expone todo esto a un usuario final. |

---

## 9. Próximo paso inmediato

Las dos preguntas que quedaban abiertas tras la búsqueda exhaustiva ya se respondieron, ambas con resultado negativo:

1. **¿La señal sobrevive a costes reales?** No. El backtest con costes de Binance (ver [4.10](#410-capa-de-evaluation--srcevaluation)) muestra que las 4 combinaciones timeframe/horizonte pierden dinero neto; en el mejor caso (15m→30min) el edge bruto por operación es ~23 veces menor que el coste de una operación de ida y vuelta.
2. **¿Un edge algo mayor, filtrando por confianza o añadiendo contexto multi-timeframe, cambia eso?** Tampoco. Ni filtrar por las predicciones de mayor magnitud ni añadir RSI/tendencia de 4h y 1d al dataset de 15m (ver [4.11](#411-contexto-multi-timeframe--intento-y-resultado)) produce un umbral con retorno neto positivo en dev, y ambas ideas se confirmaron una sola vez sobre el holdout con el mismo resultado: la mejora en accuracy direccional es real (51.6%→51.77% dev, 51.8%→51.88% holdout con contexto multi-timeframe) pero demasiado pequeña frente a un coste de 20 puntos básicos por operación. Añadir funding rate de futuros perpetuos como fuente de información distinta al precio ([4.12](#412-funding-rate-de-futuros-perpetuos--intento-y-resultado)) tampoco ayudó — ni siquiera superó el baseline en dev (51.36% vs 51.6%), así que ese intento paró ahí sin tocar el holdout.

**Conclusión:** con indicadores técnicos clásicos — solos o combinados entre timeframes — BTC/USDT a granularidad de 15-30 minutos no da suficiente edge direccional para superar comisiones estándar de Binance. No tiene sentido seguir iterando sobre este mismo tipo de features esperando que la siguiente variante cruce el umbral; construir **`src/prediction/`** sobre cualquiera de los modelos actuales serviría una estrategia que pierde dinero por diseño, no por falta de afinar. Antes de invertir más en `src/prediction/`, las palancas con más probabilidad de cambiar el panorama son de otro tipo, no más ingeniería de indicadores sobre las mismas velas:

- **Reducir frecuencia de operación de raíz**, no solo filtrar predicciones: operar en timeframes más altos donde el coste pesa menos por operación (ya se probó 1h/4h/1d en el backtest, todos también en pérdidas, pero con menos operaciones el listón para ser rentable es más bajo si el edge mejora).
- **Otra fuente de información**, no otro indicador sobre el mismo precio: datos de order book/microestructura, funding rate de futuros, o señales fuera de precio (on-chain, sentimiento) — los indicadores técnicos clásicos ya están explorados en varias combinaciones (single-timeframe y multi-timeframe) sin cruzar el umbral.
- **Replantear el objetivo**: en vez de predecir dirección para operar cada vela, usar la magnitud predicida para dimensionar posición en una estrategia de menor frecuencia, o predecir volatilidad en vez de dirección.

Si ninguna de estas resulta prometedora, la conclusión honesta del proyecto en su forma actual es que BTC/USDT a corto plazo con features de precio clásicas no es un caso de uso rentable — un resultado legítimo de la investigación, no un fallo de implementación.

**Lever 1 descartado — análisis de coste de equilibrio (`scripts/breakeven_cost_analysis.py`):** antes de invertir en la vía 2 (otra fuente de datos, la más cara de construir), se comprobó primero si simplemente pagar menos comisión bastaría. Sin reentrenar nada, se recalculó cuál sería el coste de ida y vuelta que dejaría cada estrategia exactamente en tablas (el retorno bruto medio por operación *es*, por definición, ese coste de equilibrio):

| Modelo | keep_fraction | Nº operaciones (holdout) | Coste de equilibrio |
|---|---|---|---|
| Solo 15m | 1.0 (sin filtrar) | 46.779 | 0.87 pb |
| Solo 15m | 0.05 (mejor caso) | 746 | 7.85 pb |
| 15m + contexto | 1.0 (sin filtrar) | 43.957 | 0.75 pb |
| 15m + contexto | 0.05 | 1.311 | 1.90 pb |

El esquema real de comisiones de Binance Spot (VIP0, verificado): 0.1000% maker / 0.1000% taker (20 pb ida y vuelta, el supuesto ya usado en este documento); pagando con BNB, 25% de descuento → 0.075%/0.075% (**15 pb ida y vuelta**, el mejor coste realista sin requisitos de volumen o holdings grandes). Los tiers VIP superiores bajan más, pero VIP1 ya exige 250.000 $ de volumen en 30 días o mantener 25 BNB, y los tiers realmente baratos (≈2-4 pb) exigen más de 2.000 millones $/mes — inalcanzable para este proyecto.

**Conclusión:** incluso en el escenario de coste más optimista y realista (15 pb con descuento BNB), ninguna configuración se acerca a ser rentable — el mejor caso encontrado (7.85 pb, con solo 746 operaciones en el holdout, una muestra pequeña y por tanto poco fiable) sigue siendo menos de la mitad de ese coste. Haría falta más que duplicar el edge actual del modelo para que bajar de comisión, por sí solo, cerrara la brecha. **La palanca 1 (atacar el coste) queda descartada como solución por sí sola** — no es un problema de qué tier de Binance usar, es que el edge del modelo es demasiado pequeño incluso para el mejor coste alcanzable. La vía con más recorrido real sigue siendo la 2 (otra fuente de información) o la 3 (replantear el objetivo).
