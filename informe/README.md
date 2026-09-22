# Informe

`informe.tex` — informe de la Tarea 1. Compilar **dos veces** (por el índice y las referencias):

```bash
pdflatex informe.tex && pdflatex informe.tex
```

Las figuras se leen desde `../resultados/<máquina>/`, así que hay que compilar desde
este directorio. Las figuras que todavía no existen se reemplazan solas por una caja
roja, de modo que el documento compila igual.

## Paquetes necesarios

`booktabs`, `listings`, `xcolor`, `caption`, `float`, `geometry`, `hyperref`,
`graphicx`, `amsmath`, `amssymb` y, opcionalmente, `babel-spanish`.

En Overleaf están todos. En Debian/Ubuntu:

```bash
sudo apt install texlive-latex-recommended texlive-latex-extra texlive-lang-spanish
```

Sin `babel-spanish` el documento igual compila; solo quedan en inglés los títulos
automáticos ("Contents", "Table").

## Estado

Los ítems **(a) a (i) están contestados** con los datos de la máquina 1
(`DESKTOP-52S4V2O`). Falta el ítem (j) y las columnas/figuras de la máquina 2.

Lo pendiente está marcado con cajas rojas **PENDIENTE** visibles en el PDF:

```bash
grep -n "pendiente{" informe.tex
```

Al terminar, borrar la caja de estado que está justo después del título.
