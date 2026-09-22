"""
Evidencia para el ítem (c): correctitud y reproducibilidad.

Comprueba tres cosas:
  1. Las tres versiones producen intervalos de confianza similares entre sí y
     consistentes con los coeficientes verdaderos beta*.
  2. bs_sklearn y bs_numpy son equivalentes NUMÉRICAMENTE (comparten la semilla
     42+b, así que resamplean exactamente las mismas filas). bs_auto solo puede
     ser equivalente ESTADÍSTICAMENTE: BaggingRegressor sortea con su propio RNG
     interno, así que sus resamples nunca coinciden con los de las otras dos.
  3. Reproducibilidad entre ejecuciones: repetir una versión da lo mismo.
"""
import argparse

import numpy as np

from bs_auto import run_bs_auto
from bs_numpy import ajustar_bootstrap_numpy, ajustar_bootstrap_numpy_pesos
from bs_sklearn import ajustar_bootstrap
from data_utils import generate_synthetic_data


def intervalos(beta_hats):
    return (np.percentile(beta_hats, 2.5, axis=0),
            np.percentile(beta_hats, 97.5, axis=0))


def resumen(nombre, beta_hats, beta_star):
    lo, hi = intervalos(beta_hats)
    cobertura = np.mean((lo <= beta_star) & (beta_star <= hi))
    ancho = np.mean(hi - lo)
    print(f"{nombre:<14} | beta_0 ∈ [{lo[0]:+.4f}, {hi[0]:+.4f}] "
          f"| ancho medio {ancho:.5f} | cobertura de beta*: {cobertura:.1%}")
    return lo, hi


def main(N=100000, k=300, B=48, p=None):
    from config import cores_logicos
    p = p or cores_logicos()

    from joblib import Parallel, delayed

    print(f"Parámetros: N={N}, k={k}, B={B}, p={p}\n")
    X, y, beta_star = generate_synthetic_data(N=N, k=k)

    def correr(paso):
        return np.array(Parallel(n_jobs=p)(
            delayed(paso)(X, y, 42 + b, 1) for b in range(B)
        ))

    print("=" * 78)
    print("1. Intervalos de confianza al 95 % y cobertura de beta*")
    print("=" * 78)
    bh_np = correr(ajustar_bootstrap_numpy)
    bh_sk = correr(ajustar_bootstrap)
    bh_pesos = correr(ajustar_bootstrap_numpy_pesos)
    lo_np, hi_np = resumen("bs_numpy", bh_np, beta_star)
    lo_sk, hi_sk = resumen("bs_sklearn", bh_sk, beta_star)
    lo_pe, hi_pe = resumen("bs_numpy(pesos)", bh_pesos, beta_star)

    _, bh_auto = run_bs_auto(p=p, B=B, N=N, k=k)
    lo_au, hi_au = resumen("bs_auto", bh_auto, beta_star)

    print("\n" + "=" * 78)
    print("2. Equivalencia entre versiones")
    print("=" * 78)
    d_sk = np.max(np.abs(bh_np - bh_sk))
    d_pe = np.max(np.abs(bh_np - bh_pesos))
    print(f"máx |bs_numpy - bs_sklearn|        = {d_sk:.3e}  "
          f"(mismas filas remuestreadas → deben coincidir salvo redondeo)")
    print(f"máx |bs_numpy - bs_numpy(pesos)|   = {d_pe:.3e}  "
          f"(mismo resample, distinto orden de suma en punto flotante)")
    d_au_lo = np.max(np.abs(lo_np - lo_au))
    d_au_hi = np.max(np.abs(hi_np - hi_au))
    print(f"máx dif. de extremos vs bs_auto    = {max(d_au_lo, d_au_hi):.3e}  "
          f"(RNG interno distinto → equivalencia solo estadística)")

    print("\n" + "=" * 78)
    print("3. Reproducibilidad entre ejecuciones")
    print("=" * 78)
    bh_np2 = correr(ajustar_bootstrap_numpy)
    print(f"bs_numpy repetido:  máx dif. = {np.max(np.abs(bh_np - bh_np2)):.3e} "
          f"→ {'REPRODUCIBLE' if np.allclose(bh_np, bh_np2) else 'NO reproducible'}")
    _, bh_auto2 = run_bs_auto(p=p, B=B, N=N, k=k)
    print(f"bs_auto repetido:   máx dif. = {np.max(np.abs(bh_auto - bh_auto2)):.3e} "
          f"→ {'REPRODUCIBLE' if np.allclose(bh_auto, bh_auto2) else 'NO reproducible'}")
    print("\n(bs_auto solo es reproducible porque se le fijó random_state=42;"
          "\n sin ese argumento cambia en cada ejecución.)")

    print("\n" + "=" * 78)
    print("Primeros 5 coeficientes: intervalo de bs_numpy vs beta* verdadero")
    print("=" * 78)
    print(f"{'j':<4} | {'beta*_j':>9} | {'límite inf':>11} | {'límite sup':>11} | cubre")
    for j in range(min(5, len(beta_star))):
        cubre = "sí" if lo_np[j] <= beta_star[j] <= hi_np[j] else "NO"
        print(f"{j:<4} | {beta_star[j]:>9.4f} | {lo_np[j]:>11.4f} | "
              f"{hi_np[j]:>11.4f} | {cubre}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verificación del ítem (c)")
    parser.add_argument("-N", type=int, default=100000)
    parser.add_argument("-k", type=int, default=300)
    parser.add_argument("-B", type=int, default=48)
    parser.add_argument("-p", type=int, default=None)
    args = parser.parse_args()
    main(N=args.N, k=args.k, B=args.B, p=args.p)
