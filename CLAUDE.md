# CLAUDE.md — Contexto del proyecto

## Qué es esto

Tarea 1 de **IIC3533 · Computación de Alto Rendimiento · 2026-2** (PUC).
Enunciado: [tarea01.pdf](tarea01.pdf). Entrega: **viernes 25 de septiembre de 2026, 23:59**, vía Canvas, en **formato PDF**.

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

**Máquina 1 — `DESKTOP-52S4V2O` (✅ EJECUTADA con el código corregido).**
Resultados completos en `resultados/DESKTOP-52S4V2O/`.
- WSL2, **Intel Core i5-1135G7 @ 2.40 GHz**, 4 cores físicos / **8 lógicos** → p_máx = 8
- 7.6 GiB de RAM
- conda, Python 3.13.15, NumPy 2.5.2, joblib 1.5.3, threadpoolctl 3.5.0
- BLAS = **MKL 2025.0**, 4 threads por defecto en el padre (1 dentro de cada worker con p=8)

**Máquina 2 — PENDIENTE, a cargo de un tercer integrante.**
La máquina de benja (`ALLUKA`, 24 cores) **quedó descartada por RAM**: el host tiene 16 GB
y `bs_sklearn` necesita ~1 GB por proceso, así que p alto no cabe. El entorno conda
`tarea1-hpc` quedó instalado por si se retoma.

### Resultados clave de la máquina 1

| Métrica | Valor |
|---|---|
| Mejor tiempo `bs_numpy` | 9.88 s (p=3) |
| Línea base secuencial `bs_numpy` | 11.34 s |
| **Speedup máximo** | **1.22** (`bs_sklearn`, p=3) — el problema no escala |
| Eficiencia en p=8 | 0.08–0.12 |
| Mejor (p, t) | **(4, 1)** → 11.10 s |
| Oversubscription con t=4 | **+44.6 %** de tiempo |
| Variante `--pesos` | **más lenta** (13.57 s vs 10.98 s) |

Interpretación: el problema está **limitado por ancho de banda de memoria**, no por
cómputo. Tres evidencias convergen: speedup saturado en 1.22, overhead superlineal, y
superficie T(p,t) plana (6 % de rango) entre todas las configuraciones que ocupan los 8
cores.

### Huella de memoria (medida)

| Versión | Por worker | Pico en p=24 |
|---|---|---|
| `bs_numpy` | ~360 MB | ~9.4 GB |
| `bs_sklearn` / `bs_auto` | ~1 GB | ~24 GB |

`lstsq` necesita espacio de trabajo para la SVD además de la copia de `X_b`. Por eso la
segunda máquina necesita RAM holgada, o `bs_sklearn` muere por OOM con p alto.

`joblib` con loky hace *memmap* de arreglos > 1 MB en `/dev/shm`, así que `X` se comparte
sin replicarse; lo que no se comparte es la copia por resample.

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

## Informe

`informe/informe.tex` — 19 páginas, compila sin errores. **Los ítems (a) a (i) están
contestados** con los datos de la máquina 1. Falta solo el ítem (j) y las columnas/figuras
de la máquina 2.

Lo pendiente está marcado con cajas rojas: `grep -n "pendiente{" informe/informe.tex`.
Ver [informe/README.md](informe/README.md) para compilar.

## Qué falta

1. Ejecutar `bash correr_todo.sh` en una segunda máquina con RAM suficiente.
2. Completar en el `.tex`: columna de la máquina 2 en la tabla de entorno, tablas de (f) y
   (g), figuras (reemplazar `MAQUINA2` por la etiqueta real), y redactar el ítem (j).
3. Capturas del monitor del sistema para el ítem (e).
4. Revisar la declaración de uso de IA.
