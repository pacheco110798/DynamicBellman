# Ecuación de Bellman: planificación de intervalos con pesos

Andrea Georgina Rodriguez Pacheco

Compara dos formas de calcular **OPT(j) = max(w_j + OPT(p(j)), OPT(j − 1))**: fuerza bruta y memoizado.

## Instalar

```bash
pip install -r requirements.txt
```

## Usar

| Qué | Comando |
|---|---|
| App web (https://dynamic-programming-ten.vercel.app) | `python3 -m uvicorn app:app --reload` |
| Gráficas de trabajos, árbol y comparación | `python jobs.py` |
| Solo la comparación | `python compare_opt.py` |
| Pruebas | `python -m unittest test_jobs` |
| Notebook | abrir `jobs.ipynb` |
