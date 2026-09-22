# Tarea 1 — Bootstrapping paralelo para regresión lineal

IIC3533 · Computación de Alto Rendimiento · 2026-2

## Estructura

```
.
├── correr_todo.sh          # corre la tanda completa de experimentos
├── src/                    # código
│   ├── config.py           # entorno, cores lógicos, rutas de salida por máquina
│   ├── data_utils.py       # generación de datos sintéticos          — ítem (a)
│   ├── bs_auto.py          # BaggingRegressor de sklearn             — ítem (b)
│   ├── bs_sklearn.py       # joblib.Parallel + LinearRegression      — ítem (b)
│   ├── bs_numpy.py         # joblib.Parallel + ecuación normal       — ítems (b), (e)
│   ├── verificar_correctitud.py                                      # ítem (c)
│   ├── run_experiments.py  # benchmark p = 1..p_máx                  — ítems (f), (g), (h)
│   ├── run_grid_experiment.py  # grilla (p, t)                       — ítem (i)
│   └── plot_metrics.py     # gráficos T(p), S(p), E(p), To(p)        — ítems (g), (h)
├── resultados/<máquina>/   # salidas, una carpeta por computador     — ítem (j)
├── enunciado/tarea01.pdf
├── CLAUDE.md               # contexto técnico del proyecto
└── resumen.md              # estado, errores corregidos y pendientes
```

## Cómo ejecutar

```bash
bash correr_todo.sh --prueba    # ~1 min, valida que el entorno funcione
bash correr_todo.sh             # tanda real (~35 min con 8 cores)
```

Se puede invocar desde cualquier directorio. Los resultados quedan en
`resultados/<hostname>/`; para forzar el nombre: `MAQUINA=bruno bash correr_todo.sh`.

Scripts sueltos (todos aceptan `-p`, `-B`, `-t`, `-N`, `-k`):

```bash
python src/config.py                    # entorno detectado
python src/bs_numpy.py -p 8 -t 1        # una corrida
python src/bs_numpy.py -p 8 --pesos     # variante optimizada
python src/verificar_correctitud.py     # evidencia del ítem (c)
python src/plot_metrics.py --maquina bruno
```

## Requisitos

Python 3.10+, con `numpy`, `joblib`, `scikit-learn`, `threadpoolctl` y `matplotlib`.

```bash
conda create -n tarea1-hpc python=3.13 -y
conda activate tarea1-hpc
conda install numpy matplotlib joblib threadpoolctl scikit-learn -y
```

## Estado

Ver [resumen.md](resumen.md) para el estado por ítem, los errores corregidos y lo que falta.
