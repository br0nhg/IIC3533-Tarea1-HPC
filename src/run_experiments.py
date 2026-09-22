"""
Benchmark de las tres versiones para p = 1..p_máx (ítems f, g, h).

Además de T(p) mide una línea base SECUENCIAL (un for, sin joblib, con BLAS
libre). Eso es necesario porque T(1) con joblib no es una línea base honesta:
con n_jobs=1 joblib ejecuta en el proceso padre, sin crear procesos ni
serializar los 240 MB de X, así que T(1) y T(p>=2) no miden lo mismo. El ítem
(g) pide justamente justificar la elección de T(1).
"""
import argparse
import json
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits

from config import carpeta_resultados, cores_logicos, describir_entorno, etiqueta_maquina
from data_utils import generate_synthetic_data


def bootstrap_step_numpy(X, y, semilla, t=1):
    # threadpool_limits DENTRO de la tarea: en el padre no afecta a los workers.
    with threadpool_limits(limits=t, user_api="blas"):
        np.random.seed(semilla)
        N = X.shape[0]
        indices = np.random.choice(N, size=N, replace=True)
        X_b, y_b = X[indices], y[indices]
        X_T = X_b.T
        return np.linalg.solve(X_T.dot(X_b), X_T.dot(y_b))


def bootstrap_step_sklearn(X, y, semilla, t=1):
    with threadpool_limits(limits=t, user_api="blas"):
        np.random.seed(semilla)
        N = X.shape[0]
        indices = np.random.choice(N, size=N, replace=True)
        modelo = LinearRegression(fit_intercept=False)
        modelo.fit(X[indices], y[indices])
        return modelo.coef_


def medir(descripcion, funcion):
    """
    Ejecuta una medición y devuelve el tiempo, o NaN si el sistema operativo mató
    un worker por falta de memoria.

    Es indispensable en máquinas con poca RAM: las versiones basadas en
    scikit-learn usan ~1 GB por proceso (lstsq necesita espacio de trabajo para
    la SVD además de la copia de X_b), de modo que con p alto el OOM killer las
    termina. Sin este manejo, una tanda de horas se pierde entera al primer
    fallo; con él, la celda queda en NaN y el resto continúa.
    """
    try:
        t0 = time.time()
        funcion()
        return time.time() - t0
    except Exception as e:
        nombre = type(e).__name__
        if "Terminated" in nombre or "Memory" in nombre or "Killed" in nombre:
            print(f"    [{descripcion}: sin memoria suficiente, se omite]", flush=True)
        else:
            print(f"    [{descripcion}: falló con {nombre}]", flush=True)
        return float("nan")


def medir_serial(X, y, B, paso):
    """Línea base secuencial: sin joblib y con BLAS usando todos los cores."""
    t0 = time.time()
    for b in range(B):
        paso(X, y, 42 + b, t=cores_logicos())
    return time.time() - t0


def run_benchmarks(B=48, p_max=None, t=1, saltar_sklearn=False, N=100000, k=300):
    p_max = p_max or cores_logicos()
    etiqueta = etiqueta_maquina()

    print(describir_entorno())
    print(f"\nBenchmark: B={B}, p=1..{p_max}, threads internos t={t}\n")

    X, y, _ = generate_synthetic_data(N=N, k=k)

    resultados = {
        "maquina": etiqueta,
        "cores_logicos": cores_logicos(),
        "B": B,
        "N": N,
        "k": k,
        "t_interno": t,
        "p": list(range(1, p_max + 1)),
        "auto": [],
        "sklearn": [],
        "numpy": [],
    }

    # --- Líneas base secuenciales (para justificar T(1) en el ítem g) ---
    print("Midiendo líneas base secuenciales (sin joblib, BLAS libre)...")
    resultados["serial_numpy"] = medir_serial(X, y, B, bootstrap_step_numpy)
    print(f"  serial numpy:   {resultados['serial_numpy']:.4f} s")
    if not saltar_sklearn:
        resultados["serial_sklearn"] = medir_serial(X, y, B, bootstrap_step_sklearn)
        print(f"  serial sklearn: {resultados['serial_sklearn']:.4f} s")

    print(f"\n{'p':<5} | {'bs_auto (s)':<12} | {'bs_sklearn (s)':<14} | {'bs_numpy (s)':<12}")
    print("-" * 52)

    for p in range(1, p_max + 1):
        # 1. bs_auto.py — BaggingRegressor maneja bootstrap y paralelismo internamente.
        if saltar_sklearn:
            t_auto = float("nan")
        else:
            def _auto():
                BaggingRegressor(
                    estimator=LinearRegression(fit_intercept=False),
                    n_estimators=B, n_jobs=p, bootstrap=True, max_samples=1.0,
                    random_state=42,
                ).fit(X, y)
            t_auto = medir(f"bs_auto p={p}", _auto)
        resultados["auto"].append(t_auto)

        # 2. bs_sklearn.py
        if saltar_sklearn:
            t_sk = float("nan")
        else:
            t_sk = medir(f"bs_sklearn p={p}", lambda: Parallel(n_jobs=p)(
                delayed(bootstrap_step_sklearn)(X, y, 42 + b, t) for b in range(B)))
        resultados["sklearn"].append(t_sk)

        # 3. bs_numpy.py
        t_np = medir(f"bs_numpy p={p}", lambda: Parallel(n_jobs=p)(
            delayed(bootstrap_step_numpy)(X, y, 42 + b, t) for b in range(B)))
        resultados["numpy"].append(t_np)

        print(f"{p:<5} | {t_auto:<12.4f} | {t_sk:<14.4f} | {t_np:<12.4f}", flush=True)

        # Guardado incremental tras cada p: si la tanda se interrumpe (OOM, corte
        # de luz, Ctrl-C), no se pierde lo ya medido.
        carpeta = carpeta_resultados(etiqueta)
        parcial = dict(resultados, p=list(range(1, p + 1)))
        np.save(carpeta / "benchmark_results.npy", parcial)
        with open(carpeta / "benchmark_results.json", "w") as f:
            json.dump(parcial, f, indent=2)

    carpeta = carpeta_resultados(etiqueta)
    np.save(carpeta / "benchmark_results.npy", resultados)
    with open(carpeta / "benchmark_results.json", "w") as f:
        json.dump(resultados, f, indent=2)
    print(f"\n¡Experimentos completados y guardados en {carpeta}/!")
    return resultados


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark de las tres versiones")
    parser.add_argument("-B", type=int, default=48, help="número de resamples")
    parser.add_argument("--p-max", type=int, default=None,
                        help="máximo de procesos (default: cores lógicos)")
    parser.add_argument("-t", type=int, default=1,
                        help="threads BLAS internos por proceso")
    parser.add_argument("--saltar-sklearn", action="store_true",
                        help="medir solo bs_numpy (las de sklearn son ~6x más lentas)")
    parser.add_argument("-N", type=int, default=100000, help="observaciones")
    parser.add_argument("-k", type=int, default=300, help="variables de entrada")
    args = parser.parse_args()
    run_benchmarks(B=args.B, p_max=args.p_max, t=args.t,
                   saltar_sklearn=args.saltar_sklearn, N=args.N, k=args.k)
