#!/usr/bin/env python3
"""
Prueba unitaria para verificar la carga correcta del checkpoint de Optuna.
"""

import sys
import os
from pathlib import Path
import json
import pytest

# Agregar el directorio raíz del proyecto al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modules.core.optuna_checkpoint import OptunaCheckpoint


class TestCheckpointLoading:
    """Pruebas para verificar la carga correcta del checkpoint."""
    
    def setup_method(self):
        """Configuración inicial para cada prueba."""
        self.checkpoint_dir = "checkpoints"
        self.experiment_name = "cnn2d_optuna"
        self.checkpoint = OptunaCheckpoint(
            checkpoint_dir=self.checkpoint_dir,
            experiment_name=self.experiment_name
        )
    
    def test_checkpoint_files_exist(self):
        """Verificar que los archivos del checkpoint existen."""
        checkpoint_path = Path(self.checkpoint_dir)
        
        # Verificar que el directorio existe
        assert checkpoint_path.exists(), f"Directorio {self.checkpoint_dir} no existe"
        
        # Verificar archivos específicos
        required_files = [
            "cnn2d_optuna_best_params.json",
            "cnn2d_optuna_progress.json", 
            "cnn2d_optuna_trials.json"
        ]
        
        for file_name in required_files:
            file_path = checkpoint_path / file_name
            assert file_path.exists(), f"Archivo {file_name} no existe"
            assert file_path.stat().st_size > 0, f"Archivo {file_name} está vacío"
    
    def test_best_params_loading(self):
        """Verificar que los mejores parámetros se cargan correctamente."""
        best_params_data = self.checkpoint.load_best_params()
        
        assert best_params_data is not None, "No se pudieron cargar los mejores parámetros"
        assert isinstance(best_params_data, dict), "Los mejores parámetros deben ser un diccionario"
        
        # Los parámetros están anidados bajo "best_params"
        assert "best_params" in best_params_data, "Debe contener clave 'best_params'"
        best_params = best_params_data["best_params"]
        
        # Verificar parámetros esperados
        expected_params = [
            "filters_1", "filters_2", "kernel_size_1", "kernel_size_2",
            "p_drop_conv", "p_drop_fc", "dense_units", "learning_rate",
            "weight_decay", "optimizer", "batch_size"
        ]
        
        for param in expected_params:
            assert param in best_params, f"Parámetro {param} no encontrado en mejores parámetros"
        
        # Verificar tipos de datos
        assert isinstance(best_params["filters_1"], int), "filters_1 debe ser int"
        assert isinstance(best_params["learning_rate"], float), "learning_rate debe ser float"
        assert best_params["optimizer"] in ["adam", "sgd"], "optimizer debe ser 'adam' o 'sgd'"
    
    def test_progress_loading(self):
        """Verificar que el progreso se carga correctamente."""
        progress = self.checkpoint.load_progress()
        
        assert progress is not None, "No se pudo cargar el progreso"
        assert isinstance(progress, dict), "El progreso debe ser un diccionario"
        
        # Verificar campos esperados
        expected_fields = [
            "completed_trials", "total_trials", "progress_percentage",
            "best_value", "best_trial", "timestamp"
        ]
        
        for field in expected_fields:
            assert field in progress, f"Campo {field} no encontrado en progreso"
        
        # Verificar valores lógicos
        assert progress["completed_trials"] > 0, "Debe haber trials completados"
        assert progress["total_trials"] > progress["completed_trials"], "Total debe ser mayor que completados"
        assert 0 <= progress["progress_percentage"] <= 100, "Porcentaje debe estar entre 0 y 100"
        assert progress["best_value"] > 0, "Mejor valor debe ser positivo"
    
    def test_resume_info(self):
        """Verificar que la información de reanudación es correcta."""
        resume_info = self.checkpoint.get_resume_info()
        
        assert resume_info is not None, "No se pudo obtener información de reanudación"
        assert isinstance(resume_info, dict), "La información de reanudación debe ser un diccionario"
        
        # Verificar campos esperados
        expected_fields = [
            "can_resume", "completed_trials", "total_trials", 
            "progress_percentage", "best_value", "best_trial"
        ]
        
        for field in expected_fields:
            assert field in resume_info, f"Campo {field} no encontrado en resume_info"
        
        # Verificar que se puede reanudar
        assert resume_info["can_resume"] == True, "Debe poder reanudar desde checkpoint"
        assert resume_info["completed_trials"] == 13, "Debe tener 13 trials completados"
        assert resume_info["best_trial"] == 15, "Mejor trial debe ser 15"
        assert abs(resume_info["best_value"] - 0.7313) < 0.001, "Mejor valor debe ser ~0.7313"
    
    def test_trials_loading(self):
        """Verificar que los trials se cargan correctamente."""
        trials_df = self.checkpoint.create_dataframe_from_checkpoint()
        
        assert trials_df is not None, "No se pudieron cargar los trials"
        assert len(trials_df) > 0, "Debe haber trials cargados"
        
        # Verificar columnas esperadas
        expected_columns = [
            "number", "state", "value", "f1_macro", 
            "accuracy", "precision_macro", "recall_macro"
        ]
        
        for col in expected_columns:
            assert col in trials_df.columns, f"Columna {col} no encontrada en trials"
        
        # Verificar que hay trials completados
        completed_trials = trials_df[trials_df["state"] == "COMPLETE"]
        assert len(completed_trials) == 13, "Debe haber 13 trials completados"
        
        # Verificar que el mejor trial es el 15
        best_trial = trials_df.loc[trials_df["value"].idxmax()]
        assert best_trial["number"] == 15, "Mejor trial debe ser el 15"
    
    def test_study_creation(self):
        """Verificar que se puede crear un estudio desde el checkpoint."""
        study = self.checkpoint.create_study_from_checkpoint()
        
        assert study is not None, "No se pudo crear estudio desde checkpoint"
        assert study.best_trial is not None, "El estudio debe tener un mejor trial"
        assert study.best_value is not None, "El estudio debe tener un mejor valor"
        
        # Verificar que los valores coinciden
        assert abs(study.best_value - 0.7313) < 0.001, "Mejor valor del estudio debe ser ~0.7313"
        assert study.best_trial.number == 15, "Mejor trial del estudio debe ser 15"


def test_checkpoint_integration():
    """Prueba de integración completa del checkpoint."""
    print("\n" + "="*70)
    print("PRUEBA DE INTEGRACIÓN DEL CHECKPOINT")
    print("="*70)
    
    # Crear checkpoint
    checkpoint = OptunaCheckpoint(
        checkpoint_dir="checkpoints",
        experiment_name="cnn2d_optuna"
    )
    
    # Verificar información de reanudación
    resume_info = checkpoint.get_resume_info()
    print(f"✅ Información de reanudación:")
    print(f"   - Puede reanudar: {resume_info['can_resume']}")
    print(f"   - Trials completados: {resume_info['completed_trials']}")
    print(f"   - Progreso: {resume_info['progress_percentage']:.1f}%")
    print(f"   - Mejor F1: {resume_info['best_value']:.4f}")
    print(f"   - Mejor trial: {resume_info['best_trial']}")
    
    # Verificar mejores parámetros
    best_params_data = checkpoint.load_best_params()
    best_params = best_params_data["best_params"]
    print(f"\n✅ Mejores parámetros:")
    print(f"   - Filtros: {best_params['filters_1']}, {best_params['filters_2']}")
    print(f"   - Learning rate: {best_params['learning_rate']:.2e}")
    print(f"   - Optimizador: {best_params['optimizer']}")
    print(f"   - Batch size: {best_params['batch_size']}")
    
    # Verificar trials
    trials_df = checkpoint.create_dataframe_from_checkpoint()
    print(f"\n✅ Trials cargados:")
    print(f"   - Total trials: {len(trials_df)}")
    print(f"   - Trials completados: {len(trials_df[trials_df['state'] == 'COMPLETE'])}")
    print(f"   - Trials pruned: {len(trials_df[trials_df['state'] == 'PRUNED'])}")
    
    print("\n🚀 ¡Checkpoint cargado correctamente!")
    print("="*70)


if __name__ == "__main__":
    # Ejecutar prueba de integración
    test_checkpoint_integration()
    
    # Ejecutar pruebas unitarias
    pytest.main([__file__, "-v"])
