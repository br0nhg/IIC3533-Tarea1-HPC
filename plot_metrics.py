import matplotlib
matplotlib.use('Agg') # Fuerza a Matplotlib a guardar imágenes sin abrir ventanas
import matplotlib.pyplot as plt
import numpy as np

# Cargar los datos
data = np.load("benchmark_results.npy", allow_pickle=True).item()
p = np.array(data['p'])
t_auto = np.array(data['auto'])
t_sk = np.array(data['sklearn'])
t_np = np.array(data['numpy'])

# Configuración general de los gráficos
plt.style.use('seaborn-v0_8-whitegrid')
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# ----------------- 1. Gráfico de Tiempos T(p) -----------------
axs[0].plot(p, t_auto, 'o-', label='bs_auto', color='red')
axs[0].plot(p, t_sk, 's-', label='bs_sklearn', color='orange')
axs[0].plot(p, t_np, '^-', label='bs_numpy', color='green')
axs[0].plot(p, t_np[0] / p, 'k--', label='Ideal (NumPy)')

axs[0].set_title('Tiempo de Ejecución $T(p)$', fontsize=14)
axs[0].set_xlabel('Número de procesos ($p$)', fontsize=12)
axs[0].set_ylabel('Tiempo (segundos)', fontsize=12)
axs[0].legend()

# ----------------- 2. Gráfico de Speedup S(p) -----------------
s_auto = t_auto[0] / t_auto
s_sk = t_sk[0] / t_sk
s_np = t_np[0] / t_np

axs[1].plot(p, s_auto, 'o-', label='bs_auto', color='red')
axs[1].plot(p, s_sk, 's-', label='bs_sklearn', color='orange')
axs[1].plot(p, s_np, '^-', label='bs_numpy', color='green')
axs[1].plot(p, p, 'k--', label='Ideal ($S=p$)')

axs[1].set_title('Speedup $S(p)$', fontsize=14)
axs[1].set_xlabel('Número de procesos ($p$)', fontsize=12)
axs[1].set_ylabel('Speedup', fontsize=12)
axs[1].legend()

# ----------------- 3. Gráfico de Eficiencia E(p) -----------------
e_auto = s_auto / p
e_sk = s_sk / p
e_np = s_np / p

axs[2].plot(p, e_auto, 'o-', label='bs_auto', color='red')
axs[2].plot(p, e_sk, 's-', label='bs_sklearn', color='orange')
axs[2].plot(p, e_np, '^-', label='bs_numpy', color='green')
axs[2].plot(p, np.ones_like(p), 'k--', label='Ideal ($E=1$)')

axs[2].set_title('Eficiencia $E(p)$', fontsize=14)
axs[2].set_xlabel('Número de procesos ($p$)', fontsize=12)
axs[2].set_ylabel('Eficiencia', fontsize=12)
axs[2].legend()

plt.tight_layout()
plt.savefig('graficos_metricas.png', dpi=300)
print("¡Gráficos generados y guardados exitosamente como 'graficos_metricas.png'!")

# Cálculo del Overhead To(p) = p * T(p) - T(1)
to_auto = p * t_auto - t_auto[0]
to_sk = p * t_sk - t_sk[0]
to_np = p * t_np - t_np[0]

plt.figure(figsize=(7, 5))
plt.plot(p, to_auto, 'o-', label='bs_auto', color='red')
plt.plot(p, to_sk, 's-', label='bs_sklearn', color='orange')
plt.plot(p, to_np, '^-', label='bs_numpy', color='green')

plt.title('Overhead $T_o(p) = p \cdot T(p) - T(1)$', fontsize=14)
plt.xlabel('Número de procesos ($p$)', fontsize=12)
plt.ylabel('Overhead (segundos-proceso)', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig('grafico_overhead.png', dpi=300)