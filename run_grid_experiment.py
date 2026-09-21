import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from data_utils import generate_synthetic_data

def bootstrap_step_numpy(X, y, semilla):
    np.random.seed(semilla)
    N = X.shape[0]
    indices = np.random.choice(N, size=N, replace=True)
    X_b, y_b = X[indices], y[indices]
    X_T = X_b.T
    return np.linalg.solve(X_T.dot(X_b), X_T.dot(y_b))

def run_grid():
    X, y, _ = generate_synthetic_data()
    B = 48
    p_vals = [1, 2, 4, 8]
    t_vals = [1, 2, 4, 8]
    
    grid_times = np.zeros((len(t_vals), len(p_vals)))
    
    print(f"{'t \\ p':<8} | " + " | ".join([f"p={p:<4}" for p in p_vals]))
    print("-" * 45)
    
    for i, t in enumerate(t_vals):
        row_str = f"t={t:<6} | "
        for j, p in enumerate(p_vals):
            with threadpool_limits(limits=t, user_api='blas'):
                t0 = time.time()
                Parallel(n_jobs=p)(
                    delayed(bootstrap_step_numpy)(X, y, 42 + b) for b in range(B)
                )
                grid_times[i, j] = time.time() - t0
            row_str += f"{grid_times[i, j]:<6.2f} | "
        print(row_str)

    # Graficar Mapa de Calor con Matplotlib puro
    plt.figure(figsize=(8, 6))
    plt.imshow(grid_times, cmap="YlOrRd")
    plt.colorbar(label="Tiempo (segundos)")
    plt.xticks(range(len(p_vals)), [f"p={p}" for p in p_vals])
    plt.yticks(range(len(t_vals)), [f"t={t}" for t in t_vals])

    # Agregar el texto del tiempo dentro de cada celda
    for i in range(len(t_vals)):
        for j in range(len(p_vals)):
            plt.text(j, i, f"{grid_times[i, j]:.2f}", ha="center", va="center", color="black")

    plt.title("Tiempo $T(p, t)$ en segundos (NumPy - 2D Grid)")
    plt.xlabel("Procesos ($p$)")
    plt.ylabel("Threads por proceso ($t$)")
    plt.tight_layout()
    plt.savefig("grid_heatmap.png", dpi=300)
    print("\n¡Matriz completada y mapa de calor guardado como 'grid_heatmap.png'!")

if __name__ == "__main__":
    run_grid()