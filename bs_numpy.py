import argparse
import time

import numpy as np
from joblib import Parallel, delayed
from threadpoolctl import threadpool_info, threadpool_limits

from config import cores_logicos
from data_utils import generate_synthetic_data


def ajustar_bootstrap_numpy(X, y, semilla, t=1):
    """
    Resuelve la regresión usando la ecuación normal pura con NumPy.

    IMPORTANTE: threadpool_limits se aplica ACÁ, dentro de la tarea. Cada worker
    de joblib es un proceso distinto que carga su propia copia de BLAS, así que
    un threadpool_limits puesto en el proceso padre alrededor de Parallel(...)
    no tiene ningún efecto sobre los workers cuando p >= 2.
    """
    with threadpool_limits(limits=t, user_api="blas"):
        np.random.seed(semilla)
        N = X.shape[0]

        indices = np.random.choice(N, size=N, replace=True)
        X_b = X[indices]
        y_b = y[indices]

        X_T = X_b.T
        X_T_X = X_T.dot(X_b)
        X_T_y = X_T.dot(y_b)
        return np.linalg.solve(X_T_X, X_T_y)


def ajustar_bootstrap_numpy_pesos(X, y, semilla, t=1):
    """
    Variante optimizada (ítem b): evita materializar X_b.

    X[indices] copia los 240 MB de X en cada resample con un acceso aleatorio a
    memoria. Un resample bootstrap es equivalente a pesos enteros por fila, así
    que X_b^T X_b == X^T diag(w) X, donde w cuenta cuántas veces salió cada fila.
    Se usan exactamente los mismos índices que la versión de arriba, de modo que
    el resultado es el mismo salvo por el orden de la suma en punto flotante.
    """
    with threadpool_limits(limits=t, user_api="blas"):
        np.random.seed(semilla)
        N = X.shape[0]

        indices = np.random.choice(N, size=N, replace=True)
        w = np.bincount(indices, minlength=N).astype(X.dtype)

        # Sigue habiendo un temporal del tamaño de X, pero el acceso es
        # secuencial en vez de aleatorio (mucho mejor para la caché).
        Xw = X * w[:, None]
        X_T_X = X.T.dot(Xw)
        X_T_y = X.T.dot(y * w)
        return np.linalg.solve(X_T_X, X_T_y)


def _threads_blas():
    return [(lib["internal_api"], lib["num_threads"])
            for lib in threadpool_info() if lib["user_api"] == "blas"]


def _info_threads_worker(t):
    """
    Threads BLAS vistos DESDE DENTRO de un worker (ítem e).

    Devuelve dos lecturas:
      - 'default': lo que el worker trae al arrancar. joblib/loky ya fija por su
        cuenta inner_max_num_threads = cpu_count // n_jobs, así que este valor
        no es el default global de BLAS.
      - 'limitado': lo que se ve dentro de un bloque threadpool_limits(t), que
        es lo que efectivamente usan las tareas.
    """
    np.linalg.solve(np.eye(4), np.ones(4))  # fuerza la carga de BLAS
    default = _threads_blas()
    with threadpool_limits(limits=t, user_api="blas"):
        limitado = _threads_blas()
    return default, limitado


def run_bs_numpy(p=None, B=48, t=1, pesos=False, N=100000, k=300):
    p = p or cores_logicos()
    ajustar = ajustar_bootstrap_numpy_pesos if pesos else ajustar_bootstrap_numpy

    print("=== Threads BLAS en el proceso padre (por defecto) ===")
    np.linalg.solve(np.eye(4), np.ones(4))
    for lib in threadpool_info():
        if lib["user_api"] == "blas":
            print(f"  {lib['internal_api']} {lib.get('version', '?')}: "
                  f"{lib['num_threads']} threads")
    print(f"  cores lógicos de la máquina: {cores_logicos()}")

    print(f"\nGenerando datos... (p={p}, B={B}, t={t}, "
          f"variante={'pesos' if pesos else 'índices'})")
    X, y, beta_star = generate_synthetic_data(N=N, k=k)

    # Evidencia del ítem (e): cuántos threads ve BLAS dentro de cada proceso.
    print(f"\n=== Threads BLAS vistos dentro de los workers (t={t}) ===")
    vistos = Parallel(n_jobs=p)(
        delayed(_info_threads_worker)(t) for _ in range(min(p, 4))
    )
    for i, (default, limitado) in enumerate(vistos):
        print(f"  worker {i}: al arrancar {default} → dentro de "
              f"threadpool_limits({t}) {limitado}")

    n_default = vistos[0][0][0][1] if vistos and vistos[0][0] else t
    n_limitado = vistos[0][1][0][1] if vistos and vistos[0][1] else t
    cores = cores_logicos()
    for nombre, n in (("sin limitar", n_default), (f"con t={t}", n_limitado)):
        total = p * n
        print(f"  {nombre}: {p} procesos x {n} threads = {total} sobre {cores} "
              f"cores lógicos{'  [OVERSUBSCRIPTION]' if total > cores else ''}")

    print(f"\nEjecutando bootstrapping (n_jobs={p}, threads internos={t})...")
    inicio = time.time()
    beta_hats_lista = Parallel(n_jobs=p)(
        delayed(ajustar)(X, y, semilla=42 + b, t=t) for b in range(B)
    )
    tiempo_total = time.time() - inicio

    # Paso 3 del enunciado: intervalo de confianza al 95 %.
    beta_hats = np.array(beta_hats_lista)
    limite_inferior = np.percentile(beta_hats, 2.5, axis=0)
    limite_superior = np.percentile(beta_hats, 97.5, axis=0)

    print(f"\n¡Cálculo completado en {tiempo_total:.4f} segundos!")
    print(f"Forma de la matriz de coeficientes: {beta_hats.shape} "
          f"(Esperado: {B}, {X.shape[1]})")

    cobertura = np.mean((limite_inferior <= beta_star) & (beta_star <= limite_superior))
    print(f"\nIntervalo 95% beta_0: [{limite_inferior[0]:.4f}, "
          f"{limite_superior[0]:.4f}] (Verdadero: {beta_star[0]:.4f})")
    print(f"Cobertura de beta* por los intervalos: {cobertura:.1%}")

    return tiempo_total, beta_hats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrapping OLS con NumPy puro")
    parser.add_argument("-p", type=int, default=None,
                        help="número de procesos (default: cores lógicos)")
    parser.add_argument("-B", type=int, default=48, help="número de resamples")
    parser.add_argument("-t", type=int, default=1,
                        help="threads BLAS internos por proceso")
    parser.add_argument("--pesos", action="store_true",
                        help="usar la variante optimizada con pesos multinomiales")
    parser.add_argument("-N", type=int, default=100000, help="observaciones")
    parser.add_argument("-k", type=int, default=300, help="variables de entrada")
    args = parser.parse_args()
    run_bs_numpy(p=args.p, B=args.B, t=args.t, pesos=args.pesos, N=args.N, k=args.k)
