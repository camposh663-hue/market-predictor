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

### 4.5. Tests — `tests/`

Cada módulo de negocio tiene su suite de tests con `unittest` (librería estándar de Python, sin dependencias nuevas):

- `tests/database/test_data_manager.py` — 7 tests: guardado/lectura, deduplicación, validaciones, aislamiento entre timeframes.
- `tests/database/test_parquet_repository.py` — 10 tests: persistencia real en disco (usando carpetas temporales, nunca `data/` real), fusión entre escrituras sucesivas, normalización de símbolos, creación automática de carpetas, y una prueba de integración con `DataManager`.
- `tests/sync/test_market_data_sync_service.py` — 6 tests: descarga inicial completa, reanudación incremental, fusión con lo ya guardado, skip cuando ya está al día, y validaciones. Usa un `FakeProvider` (un doble de test que cumple el contrato `BaseProvider`) para no depender de red real.

Total: **23 tests**, todos en verde. Se ejecutan con:

```bash
python -m unittest discover -s tests -v
```

### 4.6. Dependencias — `requirements.txt`

- `requests` — cliente HTTP usado por `BinanceProvider`.
- `pandas` y `pyarrow` — manejo tabular y lectura/escritura de Parquet, usados únicamente dentro de `ParquetRepository`.

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
                                Feature Engineering      ⬜ SIGUIENTE
                                          ↓
                                  Dataset Builder         ⬜
                                          ↓
                                      Training             ⬜
                                          ↓
                                    Evaluation              ⬜
                                          ↓
                                    Prediction                ⬜
                                          ↓
                                    Application                 ⬜
```

### 6.1. `src/features/` — Feature Engineering (SIGUIENTE PASO)

**Qué es:** a partir de las velas OHLCV crudas guardadas en Parquet, calcular variables (features) que un modelo de ML pueda usar para predecir: medias móviles, RSI, volatilidad, retornos pasados, volumen relativo, etc.

**Por qué es el siguiente paso lógico:** ahora mismo `data/` puede llenarse de velas reales gracias al `MarketDataSyncService`, pero esas velas crudas no sirven directamente para entrenar un modelo — hace falta transformarlas en variables predictivas.

**Diseño esperado (a validar cuando lleguemos):** funciones o clases "calculadoras de features" que reciban una serie de `MarketBar` (o el resultado de `DataManager.get_bars`) y devuelvan una tabla de features, sin conocer de dónde vinieron los datos ni cómo se van a usar después. Cada feature (media móvil, RSI...) como una unidad independiente y combinable, para poder experimentar añadiendo o quitando features sin reescribir todo.

### 6.2. `src/datasets/` — Dataset Builder

**Qué es:** combina las features generadas con la variable objetivo (el **label**): el retorno porcentual futuro sobre un horizonte configurable (ej. "cuánto sube o baja el precio en las próximas 4 horas"). Aquí se construye la tabla final `X` (features) + `y` (target) que se le da a un modelo.

**Por qué importa:** el horizonte de predicción debe ser configurable (30 min, 1h, 4h, 24h...) sin tocar código — es un requisito explícito del proyecto. También hay que evitar *data leakage* (que una feature use información del futuro sin querer).

### 6.3. `src/training/` — Entrenamiento

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
| `features/` (pendiente) | Convierte velas crudas en variables predictivas para el modelo. |
| `datasets/` (pendiente) | Combina features + retorno futuro (target) en la tabla de entrenamiento. |
| `training/` (pendiente) | Entrena modelos intercambiables sobre esa tabla. |
| `evaluation/` (pendiente) | Mide qué tan buenos son esos modelos. |
| `prediction/` (pendiente) | Sirve predicciones con un modelo ya entrenado. |
| `app/` (pendiente) | Expone todo esto a un usuario final. |

---

## 9. Próximo paso inmediato

Construir **`src/features/`**: el módulo de feature engineering que transforma las velas OHLCV crudas (ya persistibles gracias a todo lo construido hasta ahora) en variables numéricas listas para alimentar un modelo de Machine Learning. Antes de escribir código, hay que decidir y explicar: qué features se calculan primero (medias móviles, retornos, volatilidad...), cómo se representan (una clase por feature vs. funciones puras), y cómo se combinan sin acoplarse a un dataset o modelo concreto.
