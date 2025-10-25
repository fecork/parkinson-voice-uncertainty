#!/usr/bin/env python3
"""
Script simple para evaluar resultados de Talos.

Uso:
    python evaluate_talos.py results/cnn_talos_optimization/
"""

import sys
from pathlib import Path

# Agregar el directorio del proyecto al path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from modules.core.talos_evaluator import quick_evaluate


def main():
    if len(sys.argv) != 2:
        print("Uso: python evaluate_talos.py <directorio_resultados>")
        print("Ejemplo: python evaluate_talos.py results/cnn_talos_optimization/")
        sys.exit(1)

    results_dir = sys.argv[1]

    if not Path(results_dir).exists():
        print(f"ERROR: El directorio {results_dir} no existe")
        sys.exit(1)

    print(f"Evaluando resultados en: {results_dir}")
    print()

    # Evaluar resultados
    evaluation = quick_evaluate(results_dir)

    if evaluation["status"] == "ERROR":
        sys.exit(1)
    else:
        print("\nEvaluación completada exitosamente")


if __name__ == "__main__":
    main()
