"""
Optuna Checkpoint System
========================
Sistema de checkpointing automático para optimización con Optuna.
Guarda el progreso en tiempo real y permite continuar desde donde se quedó.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
import optuna
from optuna.study import Study
from optuna.trial import TrialState


class OptunaCheckpoint:
    """
    Sistema de checkpointing para optimización con Optuna.

    Guarda automáticamente:
    - Resultados de cada trial
    - Mejores parámetros encontrados
    - Estado del estudio
    - Progreso en tiempo real
    """

    def __init__(self, checkpoint_dir: str, experiment_name: str):
        """
        Args:
            checkpoint_dir: Directorio donde guardar checkpoints
            experiment_name: Nombre del experimento
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name

        # Archivos de checkpoint
        self.trials_file = self.checkpoint_dir / f"{experiment_name}_trials.json"
        self.best_params_file = (
            self.checkpoint_dir / f"{experiment_name}_best_params.json"
        )
        self.progress_file = self.checkpoint_dir / f"{experiment_name}_progress.json"
        self.study_file = self.checkpoint_dir / f"{experiment_name}_study.json"

    def save_trial(self, trial: optuna.trial.Trial, metrics: Dict[str, float]):
        """
        Guardar resultado de un trial individual.

        Args:
            trial: Trial de Optuna
            metrics: Métricas del trial
        """
        trial_data = {
            "number": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            "params": trial.params,
            "metrics": metrics,
            "datetime_start": trial.datetime_start.isoformat()
            if trial.datetime_start
            else None,
            "datetime_complete": trial.datetime_complete.isoformat()
            if trial.datetime_complete
            else None,
        }

        # Cargar trials existentes
        trials_data = self.load_trials()

        # Actualizar o agregar trial
        trials_data[trial.number] = trial_data

        # Guardar
        with open(self.trials_file, "w") as f:
            json.dump(trials_data, f, indent=2)

        print(f"💾 Trial {trial.number} guardado: F1={metrics.get('f1_macro', 0):.4f}")

    def save_best_params(self, best_params: Dict[str, Any], best_value: float):
        """
        Guardar mejores parámetros encontrados.

        Args:
            best_params: Mejores hiperparámetros
            best_value: Mejor valor encontrado
        """
        best_data = {
            "best_params": best_params,
            "best_value": best_value,
            "timestamp": pd.Timestamp.now().isoformat(),
        }

        with open(self.best_params_file, "w") as f:
            json.dump(best_data, f, indent=2)

        print(f"🏆 Mejores parámetros actualizados: F1={best_value:.4f}")

    def save_progress(
        self,
        completed_trials: int,
        total_trials: int,
        best_value: float,
        best_trial: int,
    ):
        """
        Guardar progreso general de la optimización.

        Args:
            completed_trials: Número de trials completados
            total_trials: Total de trials a ejecutar
            best_value: Mejor valor encontrado
            best_trial: Número del mejor trial
        """
        progress_data = {
            "completed_trials": completed_trials,
            "total_trials": total_trials,
            "progress_percentage": (completed_trials / total_trials) * 100,
            "best_value": best_value,
            "best_trial": best_trial,
            "timestamp": pd.Timestamp.now().isoformat(),
        }

        with open(self.progress_file, "w") as f:
            json.dump(progress_data, f, indent=2)

        print(
            f"📊 Progreso guardado: {completed_trials}/{total_trials} ({progress_data['progress_percentage']:.1f}%)"
        )

    def load_trials(self) -> Dict[int, Dict[str, Any]]:
        """Cargar trials guardados."""
        if self.trials_file.exists():
            with open(self.trials_file, "r") as f:
                return json.load(f)
        return {}

    def load_best_params(self) -> Optional[Dict[str, Any]]:
        """Cargar mejores parámetros guardados."""
        if self.best_params_file.exists():
            with open(self.best_params_file, "r") as f:
                return json.load(f)
        return None

    def load_progress(self) -> Optional[Dict[str, Any]]:
        """Cargar progreso guardado."""
        if self.progress_file.exists():
            with open(self.progress_file, "r") as f:
                return json.load(f)
        return None

    def create_study_from_checkpoint(self) -> Study:
        """
        Crear estudio de Optuna desde checkpoint.

        Returns:
            Study configurado con trials previos
        """
        # Crear estudio
        study = optuna.create_study(
            study_name=self.experiment_name,
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=5,
                interval_steps=1,
            ),
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        # Cargar trials previos
        trials_data = self.load_trials()

        if trials_data:
            print(f"🔄 Cargando {len(trials_data)} trials previos...")

            for trial_num, trial_data in trials_data.items():
                # Crear trial
                trial = study.ask()

                # Configurar parámetros
                for param, value in trial_data["params"].items():
                    if param in ["filters_1", "filters_2", "dense_units"]:
                        trial.suggest_categorical(param, [16, 32, 64, 128])
                    elif param in ["kernel_size_1", "kernel_size_2"]:
                        trial.suggest_categorical(param, [3, 5])
                    elif param == "p_drop_conv":
                        trial.suggest_float(param, 0.2, 0.5)
                    elif param == "p_drop_fc":
                        trial.suggest_float(param, 0.3, 0.6)
                    elif param == "learning_rate":
                        trial.suggest_float(param, 1e-5, 1e-2, log=True)
                    elif param == "weight_decay":
                        trial.suggest_float(param, 1e-6, 1e-3, log=True)
                    elif param == "optimizer":
                        trial.suggest_categorical(param, ["adam", "sgd"])
                    elif param == "batch_size":
                        trial.suggest_categorical(param, [16, 32, 64])

                # Reportar resultado
                if trial_data["state"] == "COMPLETE":
                    study.tell(trial, trial_data["value"])
                elif trial_data["state"] == "PRUNED":
                    study.tell(trial, trial_data["value"], state=TrialState.PRUNED)
                else:
                    study.tell(trial, trial_data["value"], state=TrialState.FAIL)

            print(f"✅ {len(trials_data)} trials cargados correctamente")

        return study

    def get_resume_info(self) -> Dict[str, Any]:
        """
        Obtener información para reanudar optimización.

        Returns:
            Dict con información de reanudación
        """
        progress = self.load_progress()
        best_params = self.load_best_params()
        trials_data = self.load_trials()

        if not progress:
            return {"can_resume": False}

        return {
            "can_resume": True,
            "completed_trials": progress["completed_trials"],
            "total_trials": progress["total_trials"],
            "progress_percentage": progress["progress_percentage"],
            "best_value": progress["best_value"],
            "best_trial": progress["best_trial"],
            "best_params": best_params["best_params"] if best_params else None,
            "trials_count": len(trials_data),
        }

    def create_dataframe_from_checkpoint(self) -> pd.DataFrame:
        """
        Crear DataFrame desde checkpoint.

        Returns:
            DataFrame con todos los trials guardados
        """
        trials_data = self.load_trials()

        if not trials_data:
            return pd.DataFrame()

        # Convertir a DataFrame
        rows = []
        for trial_num, trial_data in trials_data.items():
            row = {
                "number": trial_data["number"],
                "value": trial_data["value"],
                "state": trial_data["state"],
                **trial_data["params"],
                **trial_data["metrics"],
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def cleanup(self):
        """Limpiar archivos de checkpoint."""
        for file in [
            self.trials_file,
            self.best_params_file,
            self.progress_file,
            self.study_file,
        ]:
            if file.exists():
                file.unlink()
        print("🧹 Checkpoints limpiados")
