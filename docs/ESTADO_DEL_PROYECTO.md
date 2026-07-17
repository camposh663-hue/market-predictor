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

Total: **42 tests**, todos en verde. Se ejecutan con:

```bash
python -m unittest discover -s tests -v
```

### 4.8. Dependencias — `requirements.txt`

- `requests` — cliente HTTP usado por `BinanceProvider`.
- `pandas` y `pyarrow` — manejo tabular y lectura/escritura de Parquet, usados únicamente dentro de `ParquetRepository`.
- `pandas-ta-classic` — cálculo de indicadores técnicos, usado únicamente dentro de `TechnicalIndicatorCalculator`.
- `numpy` — cálculo de log-retornos, usado únicamente dentro de `TechnicalIndicatorCalculator`.

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
                                      Training             ⬜ SIGUIENTE
                                          ↓
                                    Evaluation              ⬜
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

**Timeframes adicionales — 4 horas y 1 día:** distinto motivo al de 15m. Los horizontes de 4h/24h ya se pueden calcular sobre velas de 1h (4 y 24 velas respectivamente) — no hacían falta velas nativas de 4h/1d para eso. La razón real para tenerlas es otra: un indicador calculado sobre velas diarias nativas (ej. RSI diario) mide tendencia macro, información distinta a la del mismo indicador sobre velas de 1h — útil para features multi-timeframe en el futuro (tendencia macro + entrada a corto plazo). No se ha construido esa capa de features todavía (sería sobre-diseñar sin haber entrenado antes un modelo base que muestre que hace falta), pero descargar las velas en sí es barato, así que se hizo ya para no depender de red más adelante: 19.524 velas de 4h y 3.257 de 1d.

**Qué queda fuera deliberadamente (no bloqueante para seguir):** un solo horizonte por llamada a `build()` — si en el futuro se quiere entrenar sobre varios horizontes a la vez, se llama a `build()` una vez por horizonte; no se ha añadido soporte multi-horizonte porque no hay evidencia todavía de que haga falta.

### 6.3. `src/training/` — Entrenamiento (SIGUIENTE PASO)

**Qué es:** el módulo que entrena modelos de ML sobre los datasets construidos. Debe soportar múltiples algoritmos intercambiables (Random Forest, XGBoost, LightGBM, CatBoost, LSTM, Transformers) sin que ninguno esté hardcodeado — es decir, otro patrón de contrato abstracto + implementaciones concretas, igual que `BaseProvider` o `BarRepository`.

**Por qué importa:** el objetivo del proyecto es poder comparar algoritmos fácilmente. Esto solo funciona si entrenar un modelo nuevo es "cambiar una pieza", no reescribir el pipeline.

### 6.4. `src/evaluation/` — Evaluación

**Qué es:** mide qué tan bien predicen los modelos entrenados: métricas de error (MAE, RMSE), métricas específicas de series temporales financieras (ej. accuracy direccional: ¿acertó si subía o bajaba?), y validación respetando el orden temporal (nunca mezclar aleatoriamente pasado y futuro al validar, porque en series de tiempo eso "hace trampa").

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
| `training/` (pendiente) | Entrena modelos intercambiables sobre esa tabla. |
| `evaluation/` (pendiente) | Mide qué tan buenos son esos modelos. |
| `prediction/` (pendiente) | Sirve predicciones con un modelo ya entrenado. |
| `app/` (pendiente) | Expone todo esto a un usuario final. |

---

## 9. Próximo paso inmediato

Construir **`src/training/`**: el módulo de entrenamiento que consume la tabla `X`/`y` producida por `DatasetBuilder` y entrena modelos de ML intercambiables (Random Forest, XGBoost, LightGBM, CatBoost, LSTM, Transformers...) sin que ninguno esté hardcodeado en el pipeline. Antes de escribir código, hay que decidir y explicar: el contrato abstracto (`ModelTrainer` o similar) que deben cumplir todas las implementaciones, cómo se hace el split train/validation/test respetando el orden temporal (nunca aleatorio, para no filtrar información futura), y con qué modelo concreto se empieza como primera implementación de referencia.
