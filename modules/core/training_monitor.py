#!/usr/bin/env python3
"""
Monitor de entrenamiento para PyTorch con Weights & Biases
=========================================================

Herramientas para monitorear el entrenamiento en tiempo real con wandb
y visualizaciones locales.
"""

import wandb
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time
from collections import defaultdict
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional
import warnings

# Suprimir warnings de wandb
warnings.filterwarnings("ignore", category=UserWarning, module="wandb")


class TrainingMonitor:
    """Monitor de entrenamiento con Weights & Biases y visualizaciones locales."""

    def __init__(
        self,
        project_name: str = "parkinson-voice-uncertainty",
        experiment_name: str = "cnn2d_training",
        config: Optional[Dict[str, Any]] = None,
        plot_every: int = 5,
        use_wandb: bool = True,
        wandb_key: Optional[str] = None,
        tags: Optional[list] = None,
        notes: Optional[str] = None,
        enable_local_plots: bool = False,
    ):
        """
        Inicializar monitor de entrenamiento.

        Args:
            project_name: Nombre del proyecto en wandb
            experiment_name: Nombre del experimento
            config: Configuración del experimento
            plot_every: Cada cuántas épocas plotear localmente
            use_wandb: Si usar Weights & Biases
            wandb_key: API key de wandb (opcional)
            tags: Tags para el experimento
            notes: Notas para el experimento
            enable_local_plots: Si generar gráficas locales (False = solo logs + W&B)
        """
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.config = config or {}
        self.plot_every = plot_every
        self.use_wandb = use_wandb
        self.enable_local_plots = enable_local_plots
        self.tags = tags or []
        self.notes = notes or ""
        self.metrics = defaultdict(list)
        self.start_time = time.time()
        self.wandb_run = None

        # Configurar wandb si está habilitado
        if self.use_wandb:
            self._setup_wandb(wandb_key)

    def _setup_wandb(self, wandb_key: Optional[str] = None):
        """Configurar Weights & Biases."""
        try:
            if wandb_key:
                wandb.login(key=wandb_key)
            else:
                wandb.login()

            self.wandb_run = wandb.init(
                project=self.project_name,
                name=self.experiment_name,
                config=self.config,
                tags=self.tags,
                notes=self.notes,
                reinit=True,
            )
            print(
                f"✅ Weights & Biases configurado: {self.project_name}/{self.experiment_name}"
            )
            print(f"   📊 URL: {self.wandb_run.url}")
        except Exception as e:
            print(f"⚠️  Error configurando wandb: {e}")
            print("   Continuando sin wandb...")
            self.use_wandb = False
            self.wandb_run = None

    def log(self, epoch: int, **kwargs):
        """Registrar métricas para una época."""
        self.metrics["epoch"].append(epoch)
        for key, value in kwargs.items():
            self.metrics[key].append(value)

        # Logging a wandb
        if self.use_wandb and self.wandb_run:
            try:
                wandb.log({"epoch": epoch, **kwargs})
            except Exception as e:
                print(f"⚠️  Error logging a wandb: {e}")

    def log_model(self, model: torch.nn.Module, input_shape: tuple = None):
        """Registrar el modelo en wandb."""
        if self.use_wandb and self.wandb_run:
            try:
                if input_shape:
                    # Crear un ejemplo de entrada
                    example_input = torch.randn(1, *input_shape)
                    wandb.watch(model, log="parameters", log_freq=10)
                else:
                    wandb.watch(model, log="parameters", log_freq=10)
                print("✅ Modelo registrado en wandb")
            except Exception as e:
                print(f"⚠️  Error registrando modelo en wandb: {e}")

    def plot_local(self, save_path: Optional[Path] = None):
        """Mostrar gráficos locales de métricas."""
        clear_output(wait=True)

        # Determinar número de subplots
        metric_keys = [k for k in self.metrics.keys() if k != "epoch"]
        n_metrics = len(metric_keys)

        if n_metrics == 0:
            print("No hay métricas para mostrar")
            return

        # Crear subplots
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + 1) // 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)

        # Plotear cada métrica
        for i, key in enumerate(metric_keys):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]

            ax.plot(self.metrics["epoch"], self.metrics[key], "b-", linewidth=2)
            ax.set_title(f"{key.replace('_', ' ').title()}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(key)
            ax.grid(True, alpha=0.3)

            # Resaltar mejor valor
            if "f1" in key.lower() or "acc" in key.lower():
                best_idx = np.argmax(self.metrics[key])
                best_epoch = self.metrics["epoch"][best_idx]
                best_value = self.metrics[key][best_idx]
                ax.plot(best_epoch, best_value, "ro", markersize=8)
                ax.annotate(
                    f"{best_value:.4f}",
                    xy=(best_epoch, best_value),
                    xytext=(10, 10),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                )

        # Ocultar subplots vacíos
        for i in range(n_metrics, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows > 1:
                axes[row, col].set_visible(False)
            else:
                axes[col].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.show()

    def should_plot(self, epoch: int):
        """Determinar si debe plotear en esta época."""
        if not self.enable_local_plots:
            return False
        return epoch % self.plot_every == 0 or epoch == 0

    def get_best_metrics(self):
        """Obtener mejores métricas hasta ahora."""
        best = {}
        for key, values in self.metrics.items():
            if key == "epoch":
                continue
            if "loss" in key.lower():
                best[f"best_{key}"] = min(values)
                best[f"best_{key}_epoch"] = self.metrics["epoch"][np.argmin(values)]
            else:
                best[f"best_{key}"] = max(values)
                best[f"best_{key}_epoch"] = self.metrics["epoch"][np.argmax(values)]
        return best

    def print_summary(self):
        """Imprimir resumen de entrenamiento."""
        elapsed = time.time() - self.start_time
        print(f"\n⏱️  Tiempo total: {elapsed / 60:.1f} minutos")

        best = self.get_best_metrics()
        print(f"\n🏆 Mejores métricas:")
        for key, value in best.items():
            if "epoch" not in key:
                print(f"   - {key}: {value:.4f}")

    def finish(self):
        """Finalizar experimento."""
        if self.use_wandb and self.wandb_run:
            try:
                wandb.finish()
                print("✅ Experimento finalizado en wandb")
            except Exception as e:
                print(f"⚠️  Error finalizando wandb: {e}")


def create_training_monitor(
    config: Dict[str, Any],
    experiment_name: str = "cnn2d_training",
    use_wandb: bool = True,
    wandb_key: Optional[str] = None,
    tags: Optional[list] = None,
    notes: Optional[str] = None,
    enable_local_plots: bool = False,
) -> TrainingMonitor:
    """
    Crear monitor de entrenamiento con configuración.

    Args:
        config: Configuración del experimento
        experiment_name: Nombre del experimento
        use_wandb: Si usar Weights & Biases
        wandb_key: API key de wandb
        tags: Tags para el experimento
        notes: Notas para el experimento
        enable_local_plots: Si generar gráficas locales (False = solo logs + W&B)

    Returns:
        TrainingMonitor configurado
    """
    return TrainingMonitor(
        project_name="parkinson-voice-uncertainty",
        experiment_name=experiment_name,
        config=config,
        use_wandb=use_wandb,
        wandb_key=wandb_key,
        tags=tags,
        notes=notes,
        enable_local_plots=enable_local_plots,
    )


def test_wandb_connection(api_key: str = None) -> bool:
    """
    Probar conexión con Weights & Biases.

    Args:
        api_key: API key de wandb

    Returns:
        True si la conexión es exitosa
    """
    try:
        if api_key:
            wandb.login(key=api_key)
        else:
            wandb.login()

        # Crear un run de prueba
        run = wandb.init(project="test-connection", name="connection-test", reinit=True)

        # Loggear algo simple
        wandb.log({"test_metric": 1.0})

        # Finalizar
        wandb.finish()

        print("✅ Conexión con Weights & Biases exitosa")
        return True

    except Exception as e:
        print(f"❌ Error conectando con Weights & Biases: {e}")
        return False
