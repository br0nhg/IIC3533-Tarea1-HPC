"""
Grilla (p, t) del ítem (i): procesos x threads BLAS internos, con p*t <= p_máx.

Dos correcciones respecto de la primera versión:
  1. threadpool_limits se aplica DENTRO de la tarea. Puesto en el proceso padre
     no afecta a los workers de loky (procesos nuevos, con su propia copia de
     BLAS), así que la grilla anterior nunca midió t.
  2. Se respeta la restricción p*t <= p_máx del enunciado. Las combinaciones
     inválidas quedan como NaN y en blanco en el mapa de calor.
"""
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from config import (carpeta_resultados, cores_logicos, describir_entorno,
                    etiqueta_maquina, valores_grilla)
from data_utils import generate_synthetic_data


def bootstrap_step_numpy(X, y, semilla, t):
    with threadpool_limits(limits=t, user_api="blas"):
        np.random.seed(semilla)
        N = X.shape[0]
        indices = np.random.choice(N, size=N, replace=True)
        X_b, y_b = X[indices], y[indices]
        X_T = X_b.T
        return np.linalg.solve(X_T.dot(X_b), X_T.dot(y_b))


def run_grid(B=48, p_max=None, N=100000, k=300):
    p_max = p_max or cores_logicos()
    etiqueta = etiqueta_maquina()

    print(describir_entorno())
    print(f"\nGrilla (p, t) con restricción p*t <= {p_max}\n")

    X, y, _ = generate_synthetic_data(N=N, k=k)
    p_vals = valores_grilla(p_max)
    t_vals = valores_grilla(p_max)

    # NaN = combinación no evaluada por violar p*t <= p_máx.
    grid_times = np.full((len(t_vals), len(p_vals)), np.nan)

    # Nota: el label se arma fuera de la f-string porque Python < 3.12 no
    # permite backslashes dentro de una f-string (PEP 701 lo habilitó en 3.12).
    encabezado = "t \\ p"
    print(f"{encabezado:<8} | " + " | ".join([f"p={p:<5}" for p in p_vals]))
    print("-" * (10 + 9 * len(p_vals)))

    for i, t in enumerate(t_vals):
        fila = f"t={t:<6} | "
        for j, p in enumerate(p_vals):
            if p * t > p_max:
                fila += f"{'--':<7}| "   # inválida: excede los cores lógicos
                continue
            try:
                t0 = time.time()
                Parallel(n_jobs=p)(
                    delayed(bootstrap_step_numpy)(X, y, 42 + b, t) for b in range(B)
                )
                grid_times[i, j] = time.time() - t0
                fila += f"{grid_times[i, j]:<6.2f} | "
            except Exception as e:
                print(f"\n    [p={p}, t={t}: falló ({type(e).__name__}), se omite]")
                fila += f"{'OOM':<7}| "
        print(fila, flush=True)

    carpeta = carpeta_resultados(etiqueta)

    # --- Mapa de calor ---
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad(color="#dddddd")  # celdas inválidas en gris
    im = ax.imshow(np.ma.masked_invalid(grid_times), cmap=cmap)
    fig.colorbar(im, ax=ax, label="Tiempo (segundos)")

    ax.set_xticks(range(len(p_vals)), [f"p={p}" for p in p_vals])
    ax.set_yticks(range(len(t_vals)), [f"t={t}" for t in t_vals])

    for i in range(len(t_vals)):
        for j in range(len(p_vals)):
            val = grid_times[i, j]
            texto = f"{val:.2f}" if np.isfinite(val) else "p·t > p máx"
            ax.text(j, i, texto, ha="center", va="center", color="black",
                    fontsize=9 if np.isfinite(val) else 7)

    ax.set_title(f"Tiempo $T(p, t)$ en segundos — {etiqueta} "
                 f"($p_{{máx}}={p_max}$, $B={B}$)", fontsize=13)
    ax.set_xlabel("Procesos ($p$)", fontsize=12)
    ax.set_ylabel("Threads BLAS por proceso ($t$)", fontsize=12)
    fig.tight_layout()
    fig.savefig(carpeta / "grid_heatmap.png", dpi=300)

    # --- Mejor combinación ---
    if np.isfinite(grid_times).any():
        i, j = np.unravel_index(np.nanargmin(grid_times), grid_times.shape)
        print(f"\nMejor combinación: p={p_vals[j]}, t={t_vals[i]} "
              f"→ {grid_times[i, j]:.2f} s")

    with open(carpeta / "grid_results.json", "w") as f:
        json.dump({"maquina": etiqueta, "p_max": p_max, "B": B,
                   "p_vals": p_vals, "t_vals": t_vals,
                   "tiempos": grid_times.tolist()}, f, indent=2)
    print(f"Mapa de calor y datos guardados en {carpeta}/")
    return grid_times


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grilla (p, t) del ítem (i)")
    parser.add_argument("-B", type=int, default=48, help="número de resamples")
    parser.add_argument("--p-max", type=int, default=None,
                        help="cores lógicos (default: os.cpu_count())")
    parser.add_argument("-N", type=int, default=100000, help="observaciones")
    parser.add_argument("-k", type=int, default=300, help="variables de entrada")
    args = parser.parse_args()
    run_grid(B=args.B, p_max=args.p_max, N=args.N, k=args.k)
