#!/usr/bin/env bash
# Corre la tanda completa de experimentos de la Tarea 1 y deja TODO registrado
# en resultados/<maquina>/terminal.txt.
#
# Se puede invocar desde cualquier directorio.
#
#   bash correr_todo.sh              # usa el hostname como etiqueta
#   MAQUINA=bruno bash correr_todo.sh
#   bash correr_todo.sh --prueba     # versión rápida (~1 min) para verificar
#                                    # que el entorno funciona antes de la real
set -u

# Trabajar siempre desde la raíz del repo, sin importar desde dónde se invoque.
cd "$(dirname "$(readlink -f "$0")")"
SRC=src

PRUEBA=""
if [ "${1:-}" = "--prueba" ]; then
    PRUEBA="-N 5000 -k 20 -B 8"
    echo ">>> MODO PRUEBA: datos reducidos, los resultados NO sirven para el informe"
fi

# En conda suele existir "python"; en un Linux pelado a veces solo "python3".
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
    if command -v python >/dev/null 2>&1; then PY=python
    elif command -v python3 >/dev/null 2>&1; then PY=python3
    else echo "ERROR: no encuentro python ni python3 en el PATH."; exit 1; fi
fi
echo ">>> Usando intérprete: $PY ($($PY --version 2>&1))"

# Chequeo de dependencias antes de gastar media hora
if ! $PY -c "import numpy, joblib, sklearn, threadpoolctl, matplotlib" 2>/dev/null; then
    echo "ERROR: faltan paquetes. Instalar con:"
    echo "  conda install numpy matplotlib joblib threadpoolctl scikit-learn -y"
    exit 1
fi

ETIQUETA="${MAQUINA:-$(hostname -s)}"
export MAQUINA="$ETIQUETA"
CARPETA="resultados/$ETIQUETA"
mkdir -p "$CARPETA"
LOG="$CARPETA/terminal.txt"

exec > >(tee "$LOG") 2>&1

echo "########################################################################"
echo "# Tarea 1 HPC — tanda completa"
echo "# Máquina: $ETIQUETA"
echo "# Fecha:   $(date -Iseconds)"
echo "########################################################################"

echo
echo "### Entorno ###"
$PY $SRC/config.py
echo
echo "--- CPU ---"
lscpu 2>/dev/null | grep -E "Model name|^CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket" \
  || sysctl -n machdep.cpu.brand_string 2>/dev/null
echo "--- Memoria ---"
free -h 2>/dev/null || vm_stat 2>/dev/null | head -4
echo "--- Python ---"
$PY --version

paso () {
    echo
    echo "########################################################################"
    echo "# $1"
    echo "########################################################################"
    shift
    local t0=$SECONDS
    if ! "$@"; then
        echo "!!! FALLÓ: $* "
        echo "!!! Abortando. Mandar este terminal.txt completo para diagnosticar."
        exit 1
    fi
    echo "[duración: $((SECONDS - t0)) s]"
}

# paso_opcional: registra el fallo pero NO aborta la tanda. Se usa en los pasos
# pesados, que pueden morir por falta de memoria con p alto y que de todos modos
# guardan resultados parciales.
paso_opcional () {
    echo
    echo "########################################################################"
    echo "# $1"
    echo "########################################################################"
    shift
    local t0=$SECONDS
    if ! "$@"; then
        echo "!!! Este paso falló, pero la tanda CONTINÚA."
        echo "!!! Causa más probable: falta de memoria con p alto."
        echo "!!! Los resultados parciales ya quedaron guardados."
    fi
    echo "[duración: $((SECONDS - t0)) s]"
}

# (c) correctitud y reproducibilidad
paso "(c) Verificación de correctitud y reproducibilidad" \
    $PY $SRC/verificar_correctitud.py $PRUEBA

# (e) oversubscription: threads vistos dentro de los workers
echo
echo "########################################################################"
echo "# (e) Oversubscription — threads BLAS dentro de los workers"
echo "# >>> DEJAR EL MONITOR DEL SISTEMA ABIERTO Y SACAR CAPTURA ACÁ <<<"
echo "########################################################################"
PMAX=$($PY -c "import sys; sys.path.insert(0, '$SRC'); from config import cores_logicos; print(cores_logicos())")
for T in 1 2 4; do
    echo
    echo "--- p=$PMAX, t=$T ---"
    $PY $SRC/bs_numpy.py -p "$PMAX" -t "$T" $PRUEBA
done

# (b) comparación de las dos variantes de bs_numpy
paso "(b) bs_numpy con índices (versión del enunciado)" \
    $PY $SRC/bs_numpy.py -p "$PMAX" -t 1 $PRUEBA
paso "(b) bs_numpy con pesos (variante optimizada)" \
    $PY $SRC/bs_numpy.py -p "$PMAX" -t 1 --pesos $PRUEBA

# (f)(g)(h) benchmark de las tres versiones
paso_opcional "(f)(g)(h) Benchmark p = 1..$PMAX de las tres versiones" \
    $PY $SRC/run_experiments.py $PRUEBA

# (i) grilla (p, t)
paso_opcional "(i) Grilla (p, t) con p*t <= $PMAX" \
    $PY $SRC/run_grid_experiment.py $PRUEBA

# (g)(h) gráficos
paso "(g)(h) Gráficos T(p), S(p), E(p) y overhead" \
    $PY $SRC/plot_metrics.py

echo
echo "########################################################################"
echo "# LISTO. Entregar la carpeta completa: $CARPETA/"
echo "########################################################################"
ls -la "$CARPETA"
