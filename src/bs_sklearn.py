import argparse
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits

from config import cores_logicos
from data_utils import generate_synthetic_data


def ajustar_bootstrap(X, y, semilla, t=1):
    """
    Tarea individual que ejecuta cada proceso.

    threadpool_limits va dentro de la tarea: cada worker de joblib es un proceso
    aparte con su propia copia de BLAS, así que limitarlo en el padre no sirve.
    """
    with threadpool_limits(limits=t, user_api="blas"):
        np.random.seed(semilla)
        N = X.shape[0]

        # (I) Sortear N índices con reemplazo
        indices = np.random.choice(N, size=N, replace=True)

        # (II) Tomar las filas correspondientes para formar X^(b) e y^(b)
        X_b = X[indices]
        y_b = y[indices]

        # (III) Calcular beta_hat
        modelo = LinearRegression(fit_intercept=False)
        modelo.fit(X_b, y_b)

        return modelo.coef_


def run_bs_sklearn(p=None, B=48, t=1, N=100000, k=300):
    p = p or cores_logicos()

    print(f"Generando datos... (p={p}, B={B}, t={t})")
    X, y, beta_star = generate_synthetic_data(N=N, k=k)

    print(f"Ejecutando Bootstrapping con joblib.Parallel (n_jobs={p})...")
    inicio = time.time()

    # Paralelismo de tareas con joblib.
    # Cada tarea recibe una semilla distinta para que los resamples difieran.
    beta_hats_lista = Parallel(n_jobs=p)(
        delayed(ajustar_bootstrap)(X, y, semilla=42 + b, t=t) for b in range(B)
    )

    tiempo_total = time.time() - inicio

    beta_hats = np.array(beta_hats_lista)
    limite_inferior = np.percentile(beta_hats, 2.5, axis=0)
    limite_superior = np.percentile(beta_hats, 97.5, axis=0)

    print(f"\n¡Cálculo completado en {tiempo_total:.4f} segundos!")
    cobertura = np.mean((limite_inferior <= beta_star) & (beta_star <= limite_superior))
    print(f"Intervalo 95% beta_0: [{limite_inferior[0]:.4f}, "
          f"{limite_superior[0]:.4f}] (Verdadero: {beta_star[0]:.4f})")
    print(f"Cobertura de beta* por los intervalos: {cobertura:.1%}")

    return tiempo_total, beta_hats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrapping OLS con sklearn + joblib")
    parser.add_argument("-p", type=int, default=None,
                        help="número de procesos (default: cores lógicos)")
    parser.add_argument("-B", type=int, default=48, help="número de resamples")
    parser.add_argument("-t", type=int, default=1,
                        help="threads BLAS internos por proceso")
    parser.add_argument("-N", type=int, default=100000, help="observaciones")
    parser.add_argument("-k", type=int, default=300, help="variables de entrada")
    args = parser.parse_args()
    run_bs_sklearn(p=args.p, B=args.B, t=args.t, N=args.N, k=args.k)
