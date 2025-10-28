#!/usr/bin/env python3
"""
Cargar resultados de Optuna desde checkpoint
===========================================

Este script carga los resultados de Optuna desde el checkpoint
y los hace disponibles para el notebook.
"""

import json
import pandas as pd
import pickle
from pathlib import Path


def load_optuna_results():
    """Cargar resultados de Optuna desde checkpoint."""

    # Cargar checkpoint
    with open("checkpoints/cnn2d_optuna_best_params.json", "r") as f:
        checkpoint = json.load(f)

    # Cargar trials CSV
    trials_df = pd.read_csv("checkpoints/cnn2d_optuna_trials.csv")

    # Crear estructura de resultados compatible
    optuna_results = {
        "best_params": checkpoint["best_params"],
        "best_value": checkpoint["best_value"],
        "best_trial": checkpoint["best_trial"],
        "results_df": trials_df,
        "analysis": {
            "best_trial": {
                "number": checkpoint["best_trial"],
                "value": checkpoint["best_value"],
                "params": checkpoint["best_params"],
            }
        },
    }

    return optuna_results


if __name__ == "__main__":
    results = load_optuna_results()
    print("✅ Resultados de Optuna cargados:")
    print(f"   - Mejor F1: {results['best_value']:.4f}")
    print(f"   - Mejor Trial: {results['best_trial']}")
    print(f"   - Trials disponibles: {len(results['results_df'])}")
