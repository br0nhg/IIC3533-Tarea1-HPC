# CLAUDE.md — Contexto del proyecto

## Qué es esto

Tarea 1 de **IIC3533 · Computación de Alto Rendimiento · 2026-2** (PUC).
Enunciado: [tarea01.pdf](enunciado/tarea01.pdf). Entrega: **viernes 25 de septiembre de 2026, 23:59**, vía Canvas, en **formato PDF**.

Tema: **bootstrapping paralelo para regresión lineal OLS** usando `joblib` (paralelismo de tareas).
Trabajo en grupo de tres personas. **Los experimentos deben correrse en al menos dos computadores distintos.**

## Parámetros fijos del experimento

| Símbolo | Valor | Significado |
|---|---|---|
| `N` | 100 000 | observaciones |
| `k` | 300 | variables de entrada (X queda de 100000×301 con la columna de unos) |
| `B` | 48 | resamples bootstrap |
| `p` | 1..p_máx | procesos (`n_jobs`), p_máx = cores lógicos de la máquina |
| `t` | threads BLAS internos | ítem (i), con restricción `p·t ≤ p_máx` |

`X` en memoria = 100000 × 301 × 8 bytes ≈ **240 MB**. Cada resample `X[indices]` crea **otra copia de 240 MB**. Este dato explica casi todo el comportamiento de escalamiento observado.

## Estructura del repositorio

| Archivo | Rol | Ítem del enunciado |
|---|---|---|
| [correr_todo.sh](correr_todo.sh) | corre la tanda completa y deja todo en `resultados/<máquina>/terminal.txt` | todos |
| [config.py](src/config.py) | etiqueta de máquina, cores lógicos, carpetas de salida, descripción del entorno | infraestructura |
| [data_utils.py](src/data_utils.py) | genera `X`, `y`, `beta_star` con semilla fija (42) | (a) |
| [bs_auto.py](src/bs_auto.py) | `BaggingRegressor` de sklearn con `n_jobs=p` | (b) |
| [bs_sklearn.py](src/bs_sklearn.py) | `joblib.Parallel` + `LinearRegression` | (b) |
| [bs_numpy.py](src/bs_numpy.py) | `joblib.Parallel` + ecuación normal; incluye variante con pesos y el diagnóstico de threads | (b), (e) |
| [verificar_correctitud.py](src/verificar_correctitud.py) | compara los intervalos de las tres versiones, la equivalencia numérica y la reproducibilidad | (c) |
| [run_experiments.py](src/run_experiments.py) | benchmark de las 3 versiones para p = 1..p_máx, más líneas base secuenciales | (f), (g) |
| [plot_metrics.py](src/plot_metrics.py) | gráficos T(p), S(p), E(p) y overhead To(p) | (g), (h) |
| [run_grid_experiment.py](src/run_grid_experiment.py) | grilla (p, t) con `p·t ≤ p_máx` → `grid_heatmap.png` | (i) |
| `resultados/<máquina>/` | salidas por máquina: `.npy`, `.json`, `.png`, logs | (j) |
| [resultados/bruno/terminal.txt](resultados/bruno/terminal.txt) | log crudo de la sesión de la máquina 1 | evidencia |

No hay tests, ni linter, ni CI. Es un repo de scripts académicos: se ejecutan a mano con `python <script>.py`.

**Todas las salidas van a `resultados/<etiqueta>/`**, donde la etiqueta es el hostname (o lo que diga la variable de entorno `MAQUINA`). Así los resultados de una máquina no pisan los de la otra, que es lo que necesita el ítem (j).

## Las dos máquinas

**Máquina 1 — "bruno" (ya ejecutada).** Todos los resultados versionados provienen de aquí.
- WSL2, usuario `bruno`, host `DESKTOP-52S4V2O`, ruta `~/universidad/IIC3533/Tareas/T1`
- **8 cores lógicos** → p_máx = 8
- Entorno conda `tareal-hpc`; BLAS = **MKL** (`libmkl_rt.so.2`), **4 threads por defecto**

**Máquina 2 — "benja" (PENDIENTE).** Esta máquina, donde corre esta sesión.
- WSL2 (`6.18.33.2-microsoft-standard-WSL2`), ruta `/home/benjasaldias/Semestres/2026-2/HPC/T1`
- CPU **Intel Core i7-14650HX**, **24 cores lógicos** (12 físicos × 2 hilos) → p_máx = 24
- Python **3.10.12 del sistema** (no conda), numpy 2.0.2, joblib 1.4.2, scikit-learn 1.5.2, threadpoolctl 3.5.0, matplotlib 3.9.2
- BLAS = **OpenBLAS 0.3.27** (`libscipy_openblas64`), **24 threads por defecto**
- ⚠️ **`/mnt/c/Users/benja/.wslconfig` limita WSL a `memory=6GB` y `swap=2GB`.** `/dev/shm` = 2.9 GB.

El contraste MKL/8 cores vs OpenBLAS/24 cores es exactamente el material del ítem (j). Vale la pena conservarlo en vez de homogeneizar los entornos.

### Bloqueador de memoria en la máquina 2

Con 6 GB, correr `p` procesos donde cada worker materializa `X_b` (240 MB) + `y_b` + el producto intermedio es inviable para `p` alto: alrededor de p ≥ 12–16 se llega a swap u OOM. Antes de ejecutar acá hay que **subir `memory` en `.wslconfig` a 12 GB y reiniciar WSL** (`wsl --shutdown` desde PowerShell), o el benchmark de p = 1..24 no termina.

`joblib` con el backend loky hace *memmap* automático de arreglos > 1 MB en `/dev/shm`, así que `X` se comparte sin replicarse; lo que no se comparte es la copia por resample.

## Cómo ejecutar

La forma normal de ejecutar es la tanda completa:

```bash
bash correr_todo.sh --prueba    # ~1 min, verifica que el entorno funciona
bash correr_todo.sh             # tanda real (~35 min con 8 cores)
```

Detecta `python`/`python3`, chequea dependencias, aborta al primer fallo y registra todo.
Scripts sueltos:

```bash
python src/config.py                # muestra el entorno detectado y dónde irán las salidas
python src/data_utils.py            # verificación de formas

# corridas sueltas (todas aceptan -p, -B, -t, -N, -k)
python src/bs_auto.py -p 8
python src/bs_sklearn.py -p 8 -t 1
python src/bs_numpy.py -p 8 -t 1            # imprime los threads vistos dentro de los workers
python src/bs_numpy.py -p 8 --pesos         # variante optimizada, sin copiar X

python src/verificar_correctitud.py         # evidencia del ítem (c)
python src/run_experiments.py               # benchmark p = 1..p_máx (usa os.cpu_count())
python src/plot_metrics.py                  # gráficos; --t1-paralelo cambia la línea base
python src/run_grid_experiment.py           # grilla (p, t) con p·t ≤ p_máx
```

Pruebas rápidas sin gastar media hora: `-N 5000 -k 20 -B 8`.
Para comparar máquinas: `python src/plot_metrics.py --maquina bruno`.

Matplotlib ya usa el backend `Agg` en los scripts de gráficos; sin eso, WSL falla con `qt.qpa.plugin: Could not load the Qt platform plugin "xcb"`.

## Convenciones

- Código, comentarios y salidas **en español**. Mantener.
- Semilla base 42; cada resample `b` usa `42 + b`.
- `fit_intercept=False` en todos los modelos sklearn, porque `X` ya trae la columna de unos.
- El intervalo de confianza al 95 % se obtiene con `np.percentile(beta_hats, [2.5, 97.5], axis=0)`.

## Problemas corregidos (22-09-2026)

Todos los bugs detectados en la revisión ya están arreglados. El diagnóstico completo de cada uno está en [resumen.md](resumen.md); lo que hay que saber para no reintroducirlos:

1. **`threadpool_limits` debe ir DENTRO de la función tarea, nunca alrededor de `Parallel(...)`.** Cada worker de loky es un proceso nuevo con su propia copia de BLAS; un límite puesto en el padre solo aplica cuando `p = 1`. Este era el bug que invalidaba los ítems (e) e (i).
2. **`BaggingRegressor` lleva `random_state=42`.** Sin eso no es reproducible.
3. **`bs_numpy.py` calcula el intervalo de confianza** y reporta la cobertura de β*.
4. **La grilla respeta `p·t ≤ p_máx`**; las celdas inválidas quedan en NaN y se pintan en gris.
5. **`p_máx` sale de `os.cpu_count()`**, no está hardcodeado en 8.
6. **Nada de backslashes dentro de f-strings.** Python 3.13 (máquina 1) los acepta desde PEP 701; Python 3.10 (máquina 2) **no compila**. Es la trampa de portabilidad más fácil de pisar entre las dos máquinas.
7. **Los títulos de matplotlib con LaTeX usan raw strings** (`r"... \cdot ..."`).

### Dato medido, útil para el informe

joblib/loky **ya mitiga parcialmente el oversubscription** por su cuenta: fija `inner_max_num_threads = cpu_count // n_jobs` en cada worker. Medido en la máquina 2 (24 cores): con `p=4` los workers arrancan con 6 threads BLAS, con `p=8` arrancan con 3. En ambos casos `p × threads = 24`, justo el número de cores. El oversubscription aparece cuando uno **sube** `t` por encima de ese default: con `p=8, t=4` son 32 threads sobre 24 cores.

## Qué falta

Estado detallado en [resumen.md](resumen.md). En una línea: **el código ya está corregido; falta subir la memoria de WSL, correr todo en la máquina 2 y escribir el informe PDF.**

⚠️ Antes de ejecutar acá: subir `memory` en `.wslconfig` a 12 GB y hacer `wsl --shutdown`. El pico medido para `p = 24` es ~9.4 GB; con los 6 GB actuales no cabe, con 12 GB sí. El host tiene 16 GB.

El enunciado permite el uso de IA **siempre que se declare en el informe**. Hay que incluir esa declaración.
