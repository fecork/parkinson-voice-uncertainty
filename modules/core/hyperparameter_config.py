"""
Sistema de Configuración de Hiperparámetros
==========================================
Permite elegir entre hiperparámetros del paper de Ibarra o los optimizados
por Optuna.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class IbarraHyperparameters:
    """Hiperparámetros exactos del paper de Ibarra 2023."""

    # Arquitectura
    kernel_size_1: int = 6
    kernel_size_2: int = 9
    depth_CL: int = 64  # depth_conv_layer
    neurons_MLP: int = 32  # dense_units
    drop_out: float = 0.2  # p_drop_conv
    neurons_MLP_D: int = 16  # Para Domain Adaptation

    # Entrenamiento
    batch_size: int = 64
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 1e-4
    n_epochs: int = 100
    early_stopping_patience: int = 15

    # Scheduler
    step_size: int = 10
    gamma: float = 0.1

    # Dropout adicional
    p_drop_fc: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para compatibilidad con CNN2D."""
        return {
            "kernel_size_1": self.kernel_size_1,
            "kernel_size_2": self.kernel_size_2,
            "filters_1": 32,  # Valor por defecto del paper
            "filters_2": self.depth_CL,
            "dense_units": self.neurons_MLP,
            "p_drop_conv": self.drop_out,
            "p_drop_fc": self.p_drop_fc,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "n_epochs": self.n_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "step_size": self.step_size,
            "gamma": self.gamma,
            "optimizer": "sgd",
            "source": "ibarra_2023_paper",
        }


class HyperparameterManager:
    """Gestor de hiperparámetros que permite elegir entre Ibarra y Optuna."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Args:
            config_path: Ruta al archivo de configuración (opcional)
        """
        self.config_path = config_path or Path("config/hyperparameter_config.json")
        self.ibarra_params = IbarraHyperparameters()

    def get_ibarra_hyperparameters(self) -> Dict[str, Any]:
        """Obtiene los hiperparámetros exactos del paper de Ibarra."""
        print("📚 Usando hiperparámetros del paper de Ibarra 2023")
        base_params = self.ibarra_params.to_dict()
        overrides = self._load_config_dict().get("ibarra_hyperparameters", {})
        return self._apply_overrides(base_params, overrides)

    def get_optuna_hyperparameters(
        self, optuna_results_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Obtiene los mejores hiperparámetros de Optuna.

        Args:
            optuna_results_path: Ruta al archivo de resultados de Optuna

        Returns:
            Diccionario con los mejores hiperparámetros de Optuna
        """
        if optuna_results_path is None:
            # Buscar automáticamente el archivo de mejores parámetros
            possible_paths = [
                Path("checkpoints/cnn2d_optuna_best_params.json"),
                Path("results/cnn_optuna_optimization/best_params.json"),
                Path("checkpoints/cnn2d_optuna_trials.json"),
            ]

            for path in possible_paths:
                if path.exists():
                    optuna_results_path = path
                    break

        if optuna_results_path is None or not optuna_results_path.exists():
            print(
                (
                    "⚠️  No se encontraron resultados de Optuna, usando "
                    "parámetros de Ibarra"
                )
            )
            return self.get_ibarra_hyperparameters()

        print(
            (
                "🔍 Cargando mejores hiperparámetros de Optuna desde: "
                f"{optuna_results_path}"
            )
        )

        try:
            with open(optuna_results_path, "r") as f:
                data = json.load(f)

            # Extraer parámetros según el formato del archivo
            if "best_params" in data:
                params = data["best_params"]
            elif "params" in data:
                params = data["params"]
            else:
                params = data

            # Asegurar que todos los parámetros necesarios estén presentes
            complete_params = self._complete_optuna_params(params)
            complete_params["source"] = "optuna_optimized"

            print("✅ Hiperparámetros de Optuna cargados exitosamente")
            return complete_params

        except Exception as e:
            print(f"❌ Error cargando parámetros de Optuna: {e}")
            print("🔄 Usando parámetros de Ibarra como fallback")
            return self.get_ibarra_hyperparameters()

    def _complete_optuna_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Completa parámetros de Optuna con valores por defecto si faltan."""
        defaults = {
            "kernel_size_1": 4,
            "kernel_size_2": 9,
            "filters_1": 128,
            "filters_2": 32,
            "dense_units": 64,
            "p_drop_conv": 0.2,
            "p_drop_fc": 0.5,
            "batch_size": 32,
            "learning_rate": 0.0005379125214937586,
            "momentum": 0.9,
            "weight_decay": 1e-4,
            "n_epochs": 100,
            "early_stopping_patience": 10,
            "step_size": 10,
            "gamma": 0.1,
            "optimizer": "sgd",
        }

        # Usar valores de Optuna si están disponibles, sino usar defaults
        complete_params = {}
        for key, default_value in defaults.items():
            complete_params[key] = params.get(key, default_value)

        return complete_params

    def _load_config_dict(self) -> Dict[str, Any]:
        """Carga el JSON de configuración o devuelve dict vacío si falla."""
        try:
            return self.load_config()
        except Exception:
            return {}

    def _apply_overrides(self, base: Dict[str, Any], overrides: Any) -> Dict[str, Any]:
        """Aplica overrides válidos sobre base, ignorando claves desconocidas."""
        if not isinstance(overrides, dict) or not overrides:
            return base
        valid = {k: v for k, v in overrides.items() if k in base}
        return {**base, **valid}

    def save_config(self, use_ibarra: bool = True, save_path: Optional[Path] = None):
        """
        Guarda la configuración actual.

        Args:
            use_ibarra: Si True, usar parámetros de Ibarra; si False, usar Optuna
            save_path: Ruta donde guardar la configuración
        """
        if save_path is None:
            save_path = self.config_path

        # Crear directorio si no existe
        save_path.parent.mkdir(parents=True, exist_ok=True)

        config = {
            "use_ibarra_hyperparameters": use_ibarra,
            "ibarra_hyperparameters": self.ibarra_params.to_dict(),
            "description": "Configuración de hiperparámetros: Ibarra (paper) vs Optuna (optimizado)",
        }

        with open(save_path, "w") as f:
            json.dump(config, f, indent=2)

        print((f"💾 Configuración guardada en: {save_path}"))

    def load_config(self, config_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Carga la configuración guardada.

        Args:
            config_path: Ruta al archivo de configuración

        Returns:
            Diccionario con la configuración cargada
        """
        if config_path is None:
            config_path = self.config_path

        if not config_path.exists():
            print((f"⚠️  Archivo de configuración no encontrado: {config_path}"))
            print("🔄 Usando configuración por defecto (Ibarra)")
            return {"use_ibarra_hyperparameters": True}

        try:
            with open(config_path, "r") as f:
                config = json.load(f)

            print((f"✅ Configuración cargada desde: {config_path}"))
            return config

        except Exception as e:
            print((f"❌ Error cargando configuración: {e}"))
            print("🔄 Usando configuración por defecto (Ibarra)")
            return {"use_ibarra_hyperparameters": True}

    def get_hyperparameters(
        self,
        use_ibarra: Optional[bool] = None,
        optuna_results_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Obtiene los hiperparámetros según la configuración.

        Args:
            use_ibarra: Si True, usar Ibarra; si False, usar Optuna; si None, cargar de config
            optuna_results_path: Ruta a resultados de Optuna

        Returns:
            Diccionario con los hiperparámetros seleccionados
        """
        if use_ibarra is None:
            # Cargar configuración guardada
            config = self.load_config()
            use_ibarra = config.get("use_ibarra_hyperparameters", True)

        if use_ibarra:
            return self.get_ibarra_hyperparameters()
        else:
            return self.get_optuna_hyperparameters(optuna_results_path)


# Función de conveniencia para uso rápido
def get_hyperparameters(
    use_ibarra: bool = True, optuna_results_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Función de conveniencia para obtener hiperparámetros.

    Args:
        use_ibarra: Si True, usar parámetros de Ibarra; si False, usar Optuna
        optuna_results_path: Ruta a resultados de Optuna

    Returns:
        Diccionario con los hiperparámetros seleccionados
    """
    manager = HyperparameterManager()
    return manager.get_hyperparameters(use_ibarra, optuna_results_path)


# Función para comparar ambos conjuntos de parámetros
def compare_hyperparameters() -> None:
    """Compara los hiperparámetros de Ibarra vs Optuna."""
    manager = HyperparameterManager()

    ibarra_params = manager.get_ibarra_hyperparameters()
    optuna_params = manager.get_optuna_hyperparameters()

    print("=" * 80)
    print("COMPARACIÓN DE HIPERPARÁMETROS: IBARRA vs OPTUNA")
    print("=" * 80)

    # Parámetros clave para comparar
    key_params = [
        "kernel_size_1",
        "kernel_size_2",
        "filters_1",
        "filters_2",
        "dense_units",
        "p_drop_conv",
        "p_drop_fc",
        "batch_size",
        "learning_rate",
        "momentum",
        "weight_decay",
    ]

    print(f"{'Parámetro':<20} {'Ibarra':<15} {'Optuna':<15} {'Diferencia':<15}")
    print("-" * 80)

    for param in key_params:
        ibarra_val = ibarra_params.get(param, "N/A")
        optuna_val = optuna_params.get(param, "N/A")

        if ibarra_val != "N/A" and optuna_val != "N/A":
            if isinstance(ibarra_val, (int, float)) and isinstance(
                optuna_val, (int, float)
            ):
                diff = optuna_val - ibarra_val
                diff_str = f"{diff:+.3f}" if isinstance(diff, float) else f"{diff:+d}"
            else:
                diff_str = "N/A"
        else:
            diff_str = "N/A"

        print(f"{param:<20} {str(ibarra_val):<15} {str(optuna_val):<15} {diff_str:<15}")

    print("=" * 80)


if __name__ == "__main__":
    # Ejemplo de uso
    print("🔧 SISTEMA DE CONFIGURACIÓN DE HIPERPARÁMETROS")
    print("=" * 60)

    # Crear manager
    manager = HyperparameterManager()

    # Mostrar parámetros de Ibarra
    print("\n📚 PARÁMETROS DE IBARRA:")
    ibarra = manager.get_ibarra_hyperparameters()
    for key, value in ibarra.items():
        print(f"  {key}: {value}")

    # Mostrar parámetros de Optuna (si están disponibles)
    print("\n🔍 PARÁMETROS DE OPTUNA:")
    optuna = manager.get_optuna_hyperparameters()
    for key, value in optuna.items():
        print(f"  {key}: {value}")

    # Comparar ambos
    print("\n📊 COMPARACIÓN:")
    compare_hyperparameters()
