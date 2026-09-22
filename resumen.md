# Resumen de estado — Tarea 1 HPC

**Fecha del resumen:** 22 de septiembre de 2026 · **Entrega:** 25 de septiembre de 2026, 23:59
**Margen real:** 3 días.

---

## 1. Puntaje: dónde estamos

| Ítem | Pts | Qué pide | Estado |
|---|---|---|---|
| (a) | 5 | Generar datos sintéticos con semilla fija | ✅ Listo |
| (b) | 10 | Tres implementaciones + comentar iteraciones de mejora | 🟡 Código listo (+ variante optimizada nueva); falta el relato |
| (c) | 10 | Correctitud y reproducibilidad | 🟡 Evidencia ya generada por `verificar_correctitud.py`; falta redactar |
| (d) | 10 | Cómo el backend multiprocessing crea procesos | 🔴 Solo redacción, nada escrito |
| (e) | 10 | Oversubscription + `threadpool_info()` | 🟡 Medición ahora sí válida; falta correr a escala real y redactar |
| (f) | 10 | Tiempos para p = 1..p_máx, tres versiones | 🟡 Hecho en máquina 1; falta máquina 2 |
| (g) | 15 | S(p), E(p), gráficos, justificar T(1) | 🟡 Línea base secuencial ya implementada; falta medirla y justificar |
| (h) | 10 | Overhead To(p) y sus fuentes | 🟡 Gráfico hecho; falta el análisis |
| (i) | 10 | Grilla (p, t) con `p·t ≤ p_máx` | 🟡 Script corregido; falta ejecutarlo |
| (j) | 10 | Comparar las dos máquinas | 🔴 Bloqueado: falta ejecutar la máquina 2 |
| — | — | Informe PDF + declaración de uso de IA | 🔴 No existe |

**El código ya no tiene bugs conocidos.** Lo que queda es ejecutar en la máquina 2 y escribir el informe: prácticamente todo el trabajo restante es medición y redacción, no programación.

---

## 2. Tareas realizadas

### Ítem (a) — Generación de datos ✅
[data_utils.py](data_utils.py) cumple al pie de la letra: `np.random.seed(42)`, luego β* ~ N(0,1) de tamaño k+1, X de N×k desde N(0,1), `np.hstack` con la columna de unos, y = Xβ* + ruido N(0,1). Formas verificadas: X (100000, 301), y (100000,), β* (301,).

### Ítem (b) — Tres implementaciones ✅ (código)
- [bs_auto.py](bs_auto.py): `BaggingRegressor(estimator=LinearRegression(fit_intercept=False), n_estimators=48, n_jobs=p, bootstrap=True, max_samples=1.0)`. Extrae los coeficientes de `estimators_` y calcula percentiles.
- [bs_sklearn.py](bs_sklearn.py): `Parallel(n_jobs=p)` sobre una tarea que sortea índices, indexa y ajusta `LinearRegression`. Calcula e imprime el intervalo.
- [bs_numpy.py](bs_numpy.py): misma estructura, pero resuelve `np.linalg.solve(XᵀX, Xᵀy)`. Además imprime `threadpool_info()`.

Las tres usan `fit_intercept=False` (correcto: X ya trae la columna de unos) y semilla `42 + b` por resample.

### Ítems (f), (g), (h) — Benchmark y gráficos 🟡 (solo máquina 1)
[run_experiments.py](run_experiments.py) corrió p = 1..8 en la máquina 1. Resultados guardados en `benchmark_results.npy`:

| p | bs_auto (s) | bs_sklearn (s) | bs_numpy (s) |
|---|---|---|---|
| 1 | 69.28 | 73.97 | 15.75 |
| 2 | 90.76 | 85.71 | 12.40 |
| 3 | 93.53 | 88.73 | **11.76** |
| 4 | 93.93 | 95.53 | 12.45 |
| 5 | 77.67 | 76.67 | 13.75 |
| 6 | 80.05 | 78.35 | 14.16 |
| 7 | 79.32 | 76.59 | 14.28 |
| 8 | 80.94 | 82.53 | 15.10 |

Hallazgo fuerte y bien fundado: **`bs_numpy` es ~5-6× más rápida que las dos versiones con sklearn.** `LinearRegression` usa `scipy.linalg.lstsq` (SVD/QR sobre la matriz completa de 100000×301), mientras que la ecuación normal factoriza una matriz de 301×301. Esto es material de primera para el ítem (b).

Gráficos generados: `graficos_metricas.png` (T, S, E en un panel de 3), `grafico_overhead.png`, `grid_heatmap.png`. Todos con título, labels, leyenda y `dpi=300`.

### Ítem (e) — Oversubscription 🟡
`threadpool_info()` en la máquina 1 reporta MKL con **4 threads por defecto**. Con p = 8 procesos eso son hasta 32 threads sobre 8 cores lógicos: oversubscription de 4×. La evidencia de diagnóstico existe; lo que falta es demostrar que la corrección funciona (ver §3.1).

### Ítem (i) — Grilla (p, t) 🟡
`run_grid_experiment.py` produjo una matriz 4×4 (`grid_heatmap.png`), pero la medición no es válida (§3.1).

---

## 3. Errores encontrados — y su corrección

Todos los bugs de esta sección **ya están arreglados** (22-09-2026). Se documentan porque varios son material directo del informe.

### 3.1 ✅ CRÍTICO — `threadpool_limits` no llegaba a los procesos worker

El patrón original era:

```python
with threadpool_limits(limits=t, user_api='blas'):
    Parallel(n_jobs=p)(delayed(tarea)(...) for b in range(B))
```

`threadpool_limits` modifica las librerías BLAS **ya cargadas en el proceso actual**. Los workers de loky son **procesos nuevos**, que cargan su propia copia de BLAS y no heredan ese límite. Para `p = 1` el límite sí aplicaba (joblib ejecuta en el proceso padre); para `p ≥ 2` **no aplicaba nada**. La grilla del ítem (i) nunca midió `t` — por eso salía plana (11.1 a 15.9 s sin estructura).

**Corregido** en `bs_numpy.py`, `bs_sklearn.py`, `run_experiments.py` y `run_grid_experiment.py`: el `with threadpool_limits(...)` va ahora **dentro de la función tarea**, que es lo que corre en el worker.

Verificado en la máquina 2 (`p=4`, 24 cores):
```
worker 0: al arrancar [('openblas', 6)] → dentro de threadpool_limits(1) [('openblas', 1)]
```

**Hallazgo adicional (va al informe).** joblib/loky **ya fija por su cuenta** `inner_max_num_threads = cpu_count // n_jobs`: con `p=4` los workers arrancan con 6 threads, con `p=8` con 3. En ambos casos `p × threads = 24` = los cores lógicos. O sea, joblib **ya mitiga el oversubscription por defecto**; el problema aparece cuando uno fuerza `t` por encima de ese valor (`p=8, t=4` → 32 threads sobre 24 cores). Esto matiza bastante la respuesta al ítem (e) y es más interesante que el diagnóstico ingenuo.

### 3.2 ✅ `bs_numpy.py` no calculaba el intervalo de confianza
El resultado de `Parallel` se asignaba a `beta_hats_lista` y ahí moría. **Corregido**: ahora calcula los percentiles 2.5/97.5, imprime el intervalo de β₀ y reporta la **cobertura de β\*** por los intervalos.

### 3.3 ✅ `BaggingRegressor` sin `random_state`
Cada ejecución sorteaba resamples distintos → no reproducible, lo contrario de lo que hay que sostener en (c). **Corregido**: `random_state=42`.

Matiz para el informe, ahora verificado con datos: aunque se fije la semilla, `bs_auto` **nunca** da los mismos resamples que las otras dos, porque usa su propio RNG interno. La equivalencia es **estadística**, no numérica. Medido con `verificar_correctitud.py`:

| Comparación | Diferencia máxima | Interpretación |
|---|---|---|
| `bs_numpy` vs `bs_sklearn` | 1.0e-14 | mismas filas remuestreadas → equivalencia **numérica** |
| `bs_numpy` vs `bs_numpy(pesos)` | 6.2e-15 | mismo resample, distinto orden de suma en punto flotante |
| `bs_numpy` vs `bs_auto` (extremos del IC) | 2.6e-2 | RNG distinto → equivalencia solo **estadística** |

Esa tabla es, tal cual, la respuesta al ítem (c).

### 3.4 ✅ T(1) no era una línea base honesta
Con `n_jobs=1` joblib ejecuta en el proceso padre — sin crear procesos, sin serializar X, con BLAS a todos los threads. Con `p ≥ 2` aparecen de golpe el memmapping de los 240 MB, el arranque de procesos y el reparto de threads. Por eso `bs_auto` y `bs_sklearn` daban speedup < 1 para toda p.

**Corregido**: `run_experiments.py` mide además una **línea base secuencial** (un `for`, sin joblib, con BLAS libre) y la guarda como `serial_numpy` / `serial_sklearn`. `plot_metrics.py` la usa por defecto para S(p) y E(p), y con `--t1-paralelo` se puede volver al criterio anterior para comparar ambos en el informe.

Para `bs_auto` no existe línea base "sin joblib" (BaggingRegressor siempre pasa por joblib; `n_jobs=1` **es** su modo secuencial), así que ahí se sigue usando T(1). Está comentado en el código y hay que decirlo en el informe.

### 3.5 ✅ La grilla no respetaba `p·t ≤ p_máx`
Recorría el producto completo `{1,2,4,8}²`, incluyendo (8,8) = 64 threads sobre 8 cores. **Corregido**: las combinaciones con `p·t > p_máx` quedan en NaN, se imprimen como `--` y salen en gris en el mapa de calor. Los valores de `p` y `t` se derivan de `p_máx` en vez de estar fijos.

### 3.6 ✅ `SyntaxWarning` en `plot_metrics.py`
`'... $p \cdot T(p)$ ...'` — `\c` no es un escape válido. **Corregido** con raw string.

### 3.7 ✅ Los resultados no estaban separados por máquina
Correr en la máquina 2 habría sobrescrito los de la máquina 1, que el ítem (j) necesita. **Corregido**: todo va a `resultados/<etiqueta>/`, con la etiqueta tomada del hostname (o de la variable de entorno `MAQUINA`). Los resultados de la máquina 1 ya están movidos a `resultados/bruno/`. `plot_metrics.py --maquina bruno` regenera sus gráficos, y sigue funcionando con el `.npy` en formato antiguo.

### 3.8 ✅ `p_max` hardcodeado en 8
**Corregido**: sale de `os.cpu_count()` vía `config.cores_logicos()`, con `--p-max` para forzarlo.

### 3.9 🔴 PENDIENTE — Límite de memoria de WSL en la máquina 2
`/mnt/c/Users/benja/.wslconfig` fija `memory=6GB`. Cada worker materializa `X_b = X[indices]`, 240 MB, más el temporal del producto. Con 24 workers eso supera de lejos los 6 GB: swap u OOM.

**Esto no se arregla desde el repo.** Hay que editar `.wslconfig` (subir `memory` a 24–32 GB) y hacer `wsl --shutdown` desde PowerShell. Es el único bloqueador que queda para ejecutar acá.

### 3.10 ✅ Incompatibilidad Python 3.13 vs 3.10 (encontrada al ejecutar)
`run_grid_experiment.py` tenía `f"{'t \\ p':<8} | "`. Python 3.13 (máquina 1) lo acepta desde PEP 701; **Python 3.10 (máquina 2) ni siquiera compila el archivo**. El script simplemente no corría acá. **Corregido** armando el label fuera de la f-string.

Vale la pena mencionarlo en el ítem (j): las diferencias entre máquinas no son solo de rendimiento.

---

## 4. Modificaciones aplicadas

### Archivos nuevos
- **`config.py`** — etiqueta de máquina, `cores_logicos()`, carpetas de salida por host, `describir_entorno()` (versión de NumPy/joblib, BLAS y threads por defecto) para dejar registrado el entorno en cada log.
- **`verificar_correctitud.py`** — evidencia del ítem (c): intervalos de las tres versiones lado a lado, cobertura de β\*, equivalencia numérica vs estadística, y reproducibilidad entre ejecuciones repitiendo cada versión.

### Archivos modificados
| Archivo | Cambios |
|---|---|
| `bs_numpy.py` | `threadpool_limits` dentro de la tarea · cálculo del IC y cobertura · diagnóstico de threads desde dentro de los workers · variante `--pesos` · argparse |
| `bs_sklearn.py` | `threadpool_limits` dentro de la tarea · cobertura · argparse |
| `bs_auto.py` | `random_state=42` · cobertura · argparse |
| `run_experiments.py` | `p_máx` dinámico · líneas base secuenciales · `threadpool_limits` dentro de la tarea · salida por máquina (`.npy` + `.json`) · `--saltar-sklearn` |
| `run_grid_experiment.py` | restricción `p·t ≤ p_máx` con celdas enmascaradas · `threadpool_limits` dentro de la tarea · valores derivados de `p_máx` · arreglo de f-string · reporta la mejor combinación |
| `plot_metrics.py` | raw string · línea base secuencial para S(p)/E(p) con `--t1-paralelo` · carga por máquina · tabla de resumen para el informe |

Todo se validó con corridas reducidas (`-N 5000 -k 20 -B 12`). Los gráficos de la máquina 1 se regeneraron con el script corregido y siguen cuadrando.

### Optimización agregada para el ítem (b)

El enunciado pide explícitamente iterar y comentar las mejoras. La primera (ecuación normal en vez de `lstsq`) da el 5-6× y está documentada en `terminal.txt`. Se agregó una segunda como `ajustar_bootstrap_numpy_pesos` (flag `--pesos`):

`X_b = X[indices]` copia 240 MB por resample con **acceso aleatorio** a memoria, y es el cuello de botella de ancho de banda que explica por qué agregar procesos deja de ayudar. Un resample bootstrap equivale a pesos enteros por fila, así que `X_bᵀX_b == Xᵀ diag(w) X`:

```python
indices = np.random.choice(N, size=N, replace=True)
w = np.bincount(indices, minlength=N).astype(X.dtype)
Xw = X * w[:, None]          # acceso secuencial, no aleatorio
beta = np.linalg.solve(X.T.dot(Xw), X.T.dot(y * w))
```

Usa **los mismos índices** que la versión original, así que el resultado coincide (verificado: diferencia 6.2e-15, solo orden de suma en punto flotante). Sigue habiendo un temporal del tamaño de X, pero el patrón de acceso es secuencial. **Falta medir si gana en la corrida completa** — y tanto si gana como si no, medirlo y comentarlo es exactamente lo que pide el ítem.

---

## 5. Pendientes para cerrar la tarea

### Ejecución en la máquina 2 (esta)
- [ ] **Subir `memory` en `.wslconfig` a 24–32 GB + `wsl --shutdown`** ← único bloqueador
- [ ] `python config.py` para dejar registrado el entorno
- [ ] `python verificar_correctitud.py` con los parámetros reales (N=100000, k=300, B=48)
- [ ] `python run_experiments.py` con p = 1..24
- [ ] `python run_grid_experiment.py` (grilla ya restringida a `p·t ≤ 24`)
- [ ] `python bs_numpy.py -p 24 -t 1` observando el Monitor del Sistema (ítem e)
- [ ] Comparar `--pesos` contra la versión con índices en la corrida completa (ítem b)
- [ ] `python plot_metrics.py`
- [ ] Guardar el log de la sesión en `resultados/<máquina>/terminal.txt`

**Nota sobre el entorno:** esta máquina usa Python 3.10 del sistema con OpenBLAS; la máquina 1 usa conda con Python 3.13 y MKL. **No homogeneizar** — esa diferencia es el contenido del ítem (j).

### Informe PDF
- [ ] (b) Iteraciones de optimización: `lstsq` → ecuación normal (5-6×), y luego índices → pesos
- [ ] (c) Tabla de equivalencia de §3.3 + cobertura de β\* + condiciones de reproducibilidad (misma semilla, mismo orden de reducción en punto flotante, `random_state` fijo, y que el número de workers no altere el resultado porque cada tarea lleva su propia semilla)
- [ ] (d) `fork` vs `spawn`, copy-on-write, memmapping de joblib para arreglos > 1 MB en `/dev/shm`, y qué pasa con X e y al lanzar p procesos — **completamente por escribir**
- [ ] (e) Oversubscription: incluir el hallazgo de que joblib ya fija `cpu_count // n_jobs` (§3.1), con las lecturas de `threadpool_info()` desde dentro de los workers
- [ ] (f) Tabla/gráfico de las dos máquinas
- [ ] (g) S(p), E(p) + justificación de T(1): mostrar ambas líneas base y explicar por qué la secuencial es la honesta
- [ ] (h) Fuentes de overhead: creación de procesos, memmap de los 240 MB, la copia por resample, contención de ancho de banda, desbalance con B=48
- [ ] (i) Heatmap válido + mejor (p, t)
- [ ] (j) 8 vs 24 cores lógicos, MKL vs OpenBLAS, Python 3.13 vs 3.10 (§3.10), techos de memoria distintos
- [ ] **Declaración de uso de IA** (el enunciado lo permite pero exige declararlo)
- [ ] Revisar legibilidad de gráficos: títulos, labels, leyendas, tamaño de fuente (se evalúa)

### Nota sobre B = 48 y p = 24
Con p = 24 y B = 48 tocan exactamente 2 tareas por worker: el reparto es perfecto, pero el granulado es muy grueso y cualquier worker lento domina el tiempo total. Mencionarlo al analizar el overhead y la caída de eficiencia con p alto.
