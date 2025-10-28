#!/usr/bin/env python3
"""
Script para integrar automáticamente el sistema de hiperparámetros en el notebook.
"""

import json
import re
from pathlib import Path


def read_notebook_cell(file_path, cell_content):
    """Lee el contenido de una celda del notebook."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def create_notebook_cell(cell_content, cell_type="code"):
    """Crea una celda de notebook en formato JSON."""
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": cell_content.split("\n")
        if isinstance(cell_content, str)
        else cell_content,
        "execution_count": None,
        "outputs": [],
    }


def integrate_hyperparameter_system():
    """Integra el sistema de hiperparámetros en el notebook."""

    # Rutas de archivos
    notebook_path = Path("research/cnn2d_training.ipynb")
    selector_path = Path("research/notebook_cell_selector.py")
    optuna_path = Path("research/notebook_cell_optuna_replacement.py")
    model_path = Path("research/notebook_cell_model_creation.py")
    training_path = Path("research/notebook_cell_training_config.py")

    # Verificar que los archivos existen
    if not notebook_path.exists():
        print(f"❌ No se encontró el notebook: {notebook_path}")
        return

    if not all(
        [
            selector_path.exists(),
            optuna_path.exists(),
            model_path.exists(),
            training_path.exists(),
        ]
    ):
        print("❌ No se encontraron todos los archivos de celdas necesarios")
        return

    # Leer el notebook actual
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    # Leer contenidos de las celdas
    selector_content = read_notebook_cell(selector_path, "")
    optuna_content = read_notebook_cell(optuna_path, "")
    model_content = read_notebook_cell(model_path, "")
    training_content = read_notebook_cell(training_path, "")

    print("🔧 Integrando sistema de hiperparámetros en el notebook...")

    # 1. Insertar celda del selector después de la celda de configuración
    selector_cell = create_notebook_cell(selector_content)

    # Buscar la celda de configuración (Cell 3)
    config_cell_index = None
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code" and "CONFIGURACIÓN" in "".join(
            cell.get("source", [])
        ):
            config_cell_index = i
            break

    if config_cell_index is not None:
        # Insertar después de la celda de configuración
        notebook["cells"].insert(config_cell_index + 1, selector_cell)
        print("✅ Celda del selector insertada")
    else:
        print("⚠️  No se encontró la celda de configuración, insertando al principio")
        notebook["cells"].insert(0, selector_cell)

    # 2. Reemplazar celda de Optuna
    optuna_replaced = False
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code" and "OPTUNA" in "".join(cell.get("source", [])):
            notebook["cells"][i] = create_notebook_cell(optuna_content)
            optuna_replaced = True
            break

    if optuna_replaced:
        print("✅ Celda de Optuna reemplazada")
    else:
        print("⚠️  No se encontró la celda de Optuna")

    # 3. Reemplazar celda de creación del modelo
    model_replaced = False
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code" and "CNN2D(" in "".join(cell.get("source", [])):
            notebook["cells"][i] = create_notebook_cell(model_content)
            model_replaced = True
            break

    if model_replaced:
        print("✅ Celda de creación del modelo reemplazada")
    else:
        print("⚠️  No se encontró la celda de creación del modelo")

    # 4. Reemplazar celda de configuración de entrenamiento
    training_replaced = False
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code" and "FINAL_TRAINING_CONFIG" in "".join(
            cell.get("source", [])
        ):
            notebook["cells"][i] = create_notebook_cell(training_content)
            training_replaced = True
            break

    if training_replaced:
        print("✅ Celda de configuración de entrenamiento reemplazada")
    else:
        print("⚠️  No se encontró la celda de configuración de entrenamiento")

    # Guardar el notebook modificado
    backup_path = notebook_path.with_suffix(".ipynb.backup")
    notebook_path.rename(backup_path)
    print(f"💾 Backup creado: {backup_path}")

    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)

    print(f"✅ Notebook actualizado: {notebook_path}")
    print("\n🎯 PRÓXIMOS PASOS:")
    print("1. Abre el notebook actualizado")
    print("2. Ejecuta la celda del selector de hiperparámetros")
    print("3. Cambia USE_IBARRA_HYPERPARAMETERS = True/False según quieras")
    print("4. Ejecuta el resto del notebook normalmente")


if __name__ == "__main__":
    integrate_hyperparameter_system()
