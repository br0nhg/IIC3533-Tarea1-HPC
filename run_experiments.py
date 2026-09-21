import time
import numpy as np
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
from data_utils import generate_synthetic_data

def bootstrap_step_numpy(X, y, semilla):
    np.random.seed(semilla)
    N = X.shape[0]
    indices = np.random.choice(N, size=N, replace=True)
    X_b, y_b = X[indices], y[indices]
    X_T = X_b.T
    return np.linalg.solve(X_T.dot(X_b), X_T.dot(y_b))

def bootstrap_step_sklearn(X, y, semilla):
    np.random.seed(semilla)
    N = X.shape[0]
    indices = np.random.choice(N, size=N, replace=True)
    modelo = LinearRegression(fit_intercept=False)
    modelo.fit(X[indices], y[indices])
    return modelo.coef_

def run_benchmarks(B=48, p_max=8):
    X, y, _ = generate_synthetic_data()
    
    resultados = {
        'p': list(range(1, p_max + 1)),
        'auto': [],
        'sklearn': [],
        'numpy': []
    }

    print(f"{'p':<5} | {'bs_auto (s)':<12} | {'bs_sklearn (s)':<14} | {'bs_numpy (s)':<12}")
    print("-" * 50)

    for p in range(1, p_max + 1):
        with threadpool_limits(limits=1, user_api='blas'):
            # 1. bs_auto.py
            t0 = time.time()
            bagging = BaggingRegressor(
                estimator=LinearRegression(fit_intercept=False),
                n_estimators=B, n_jobs=p, bootstrap=True, max_samples=1.0
            )
            bagging.fit(X, y)
            t_auto = time.time() - t0
            resultados['auto'].append(t_auto)

            # 2. bs_sklearn.py
            t0 = time.time()
            Parallel(n_jobs=p)(
                delayed(bootstrap_step_sklearn)(X, y, 42 + b) for b in range(B)
            )
            t_sk = time.time() - t0
            resultados['sklearn'].append(t_sk)

            # 3. bs_numpy.py
            t0 = time.time()
            Parallel(n_jobs=p)(
                delayed(bootstrap_step_numpy)(X, y, 42 + b) for b in range(B)
            )
            t_np = time.time() - t0
            resultados['numpy'].append(t_np)

        print(f"{p:<5} | {t_auto:<12.4f} | {t_sk:<14.4f} | {t_np:<12.4f}")

    # Guardar resultados en un archivo .npy para usarlos en los gráficos
    np.save("benchmark_results.npy", resultados)
    print("\n¡Experimentos completados y guardados en 'benchmark_results.npy'!")

if __name__ == "__main__":
    run_benchmarks()