"""
Pipeline de entrenamiento de CNN con estimación de incertidumbre.

Este script replica el notebook cnn_uncertainty_training.ipynb
para poder ejecutarse desde línea de comandos.

Uso:
    python pipelines/train_cnn_uncertainty.py
"""

import sys
from pathlib import Path
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Agregar módulos al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.augmentation import create_augmented_dataset
from modules.dataset import to_pytorch_tensors
from modules.uncertainty_model import UncertaintyCNN, print_uncertainty_model_summary
from modules.uncertainty_training import (
    train_uncertainty_model,
    evaluate_with_uncertainty,
    print_uncertainty_results,
)
from modules.uncertainty_visualization import (
    plot_uncertainty_histograms,
    plot_reliability_diagram,
    plot_uncertainty_scatter,
    plot_training_history_uncertainty,
)
from modules.cnn_utils import plot_confusion_matrix


class DictDataset(Dataset):
    """Wrapper para convertir tensores en diccionarios."""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {"spectrogram": self.X[idx], "label": self.y[idx]}


def main():
    """Pipeline principal."""

    # ============================================================
    # CONFIGURACIÓN
    # ============================================================
    print("=" * 70)
    print("🧠 PIPELINE DE ENTRENAMIENTO CON INCERTIDUMBRE")
    print("=" * 70)

    # Paths
    DATA_PATH_HEALTHY = "./data/vowels_healthy"
    DATA_PATH_PARKINSON = "./data/vowels_pk"
    CACHE_DIR_HEALTHY = "./cache/healthy"
    CACHE_DIR_PARKINSON = "./cache/parkinson"

    # Augmentation config
    AUGMENTATION_TYPES = ["original", "pitch_shift", "time_stretch", "noise"]
    NUM_SPEC_AUGMENT_VERSIONS = 2

    # Model config
    N_CLASSES = 2
    DROPOUT_P = 0.25
    S_MIN = -10.0
    S_MAX = 3.0
    INPUT_SHAPE = (65, 41)

    # Training config
    N_EPOCHS = 60
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    T_NOISE = 5
    EARLY_STOPPING_PATIENCE = 15

    # Inference config
    T_TEST = 30

    # Output
    SAVE_DIR = Path("./results/cnn_uncertainty")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    print(f"\n📋 Configuración:")
    print(f"   • Device: {device}")
    print(f"   • Batch size: {BATCH_SIZE}")
    print(f"   • Dropout: {DROPOUT_P}")
    print(f"   • T_noise: {T_NOISE}")
    print(f"   • T_test: {T_TEST}")
    print(f"   • Epochs: {N_EPOCHS}")
    print("=" * 70)

    # ============================================================
    # CARGA DE DATOS
    # ============================================================
    print("\n📁 CARGANDO DATOS...")

    # Healthy
    audio_files_healthy = list(Path(DATA_PATH_HEALTHY).glob("*.egg"))
    augmented_dataset_healthy = create_augmented_dataset(
        audio_files_healthy,
        augmentation_types=AUGMENTATION_TYPES,
        apply_spec_augment=True,
        num_spec_augment_versions=NUM_SPEC_AUGMENT_VERSIONS,
        use_cache=True,
        cache_dir=CACHE_DIR_HEALTHY,
        force_regenerate=False,
        progress_every=5,
    )
    X_healthy, _, _, _ = to_pytorch_tensors(augmented_dataset_healthy)

    # Parkinson
    audio_files_parkinson = list(Path(DATA_PATH_PARKINSON).glob("*.egg"))
    augmented_dataset_parkinson = create_augmented_dataset(
        audio_files_parkinson,
        augmentation_types=AUGMENTATION_TYPES,
        apply_spec_augment=True,
        num_spec_augment_versions=NUM_SPEC_AUGMENT_VERSIONS,
        use_cache=True,
        cache_dir=CACHE_DIR_PARKINSON,
        force_regenerate=False,
        progress_every=5,
    )
    X_parkinson, _, _, _ = to_pytorch_tensors(augmented_dataset_parkinson)

    # Combinar
    X_combined = torch.cat([X_healthy, X_parkinson], dim=0)
    y_combined = torch.cat(
        [
            torch.zeros(len(X_healthy), dtype=torch.long),
            torch.ones(len(X_parkinson), dtype=torch.long),
        ],
        dim=0,
    )

    print("Datos cargados: {} muestras".format(len(X_combined)))

    # ============================================================
    # SPLIT
    # ============================================================
    print("\n📊 CREANDO SPLITS...")

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_combined, y_combined, test_size=0.15, random_state=42, stratify=y_combined
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.176, random_state=42, stratify=y_train_val
    )

    print(
        "Train: {} | Val: {} | Test: {}".format(len(X_train), len(X_val), len(X_test))
    )

    # ============================================================
    # DATALOADERS
    # ============================================================
    print("\n📦 CREANDO DATALOADERS...")

    train_dataset = DictDataset(X_train, y_train)
    val_dataset = DictDataset(X_val, y_val)
    test_dataset = DictDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0
    )

    print("DataLoaders creados")

    # ============================================================
    # MODELO
    # ============================================================
    print("\n🏗️  CREANDO MODELO...")

    model = UncertaintyCNN(
        n_classes=N_CLASSES,
        p_drop_conv=DROPOUT_P,
        p_drop_fc=DROPOUT_P,
        input_shape=INPUT_SHAPE,
        s_min=S_MIN,
        s_max=S_MAX,
    ).to(device)

    print_uncertainty_model_summary(model)

    # ============================================================
    # ENTRENAMIENTO
    # ============================================================
    print("\n🚀 INICIANDO ENTRENAMIENTO...")

    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    training_results = train_uncertainty_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        n_epochs=N_EPOCHS,
        n_noise_samples=T_NOISE,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        save_dir=SAVE_DIR,
        verbose=True,
    )

    model = training_results["model"]
    history = training_results["history"]
    best_val_loss = training_results["best_val_loss"]

    print("Entrenamiento completado")
    print("   Mejor val_loss: {:.4f}".format(best_val_loss))

    # ============================================================
    # EVALUACIÓN CON MC DROPOUT
    # ============================================================
    print("\n🎯 EVALUACIÓN CON INCERTIDUMBRE...")

    test_metrics = evaluate_with_uncertainty(
        model=model,
        loader=test_loader,
        device=device,
        n_mc_samples=T_TEST,
        class_names=["Healthy", "Parkinson"],
    )

    print_uncertainty_results(test_metrics, class_names=["Healthy", "Parkinson"])

    # Guardar métricas
    metrics_to_save = {
        "accuracy": float(test_metrics["accuracy"]),
        "precision": float(test_metrics["precision"]),
        "recall": float(test_metrics["recall"]),
        "f1": float(test_metrics["f1"]),
        "nll": float(test_metrics["nll"]),
        "brier": float(test_metrics["brier"]),
        "ece": float(test_metrics["ece"]),
        "mean_entropy": float(test_metrics["mean_entropy"]),
        "mean_epistemic": float(test_metrics["mean_epistemic"]),
        "mean_aleatoric": float(test_metrics["mean_aleatoric"]),
        "entropy_correct": float(test_metrics["entropy_correct"]),
        "entropy_incorrect": float(test_metrics["entropy_incorrect"]),
        "epistemic_correct": float(test_metrics["epistemic_correct"]),
        "epistemic_incorrect": float(test_metrics["epistemic_incorrect"]),
    }

    with open(SAVE_DIR / "test_metrics_uncertainty.json", "w") as f:
        json.dump(metrics_to_save, f, indent=2)

    print("\n💾 Métricas guardadas")

    # ============================================================
    # VISUALIZACIONES
    # ============================================================
    print("\n📊 GENERANDO VISUALIZACIONES...")

    # Historial
    plot_training_history_uncertainty(
        history, save_path=SAVE_DIR / "training_history.png", show=False
    )

    # Histogramas
    plot_uncertainty_histograms(
        test_metrics, save_path=SAVE_DIR / "uncertainty_histograms.png", show=False
    )

    # Reliability
    plot_reliability_diagram(
        test_metrics,
        n_bins=10,
        save_path=SAVE_DIR / "reliability_diagram.png",
        show=False,
    )

    # Scatter
    plot_uncertainty_scatter(
        test_metrics, save_path=SAVE_DIR / "uncertainty_scatter.png", show=False
    )

    # Confusión
    cm = confusion_matrix(test_metrics["targets"], test_metrics["predictions"])
    plot_confusion_matrix(
        cm,
        class_names=["Healthy", "Parkinson"],
        title="Matriz de Confusión - Test Set (CNN con Incertidumbre)",
        save_path=SAVE_DIR / "confusion_matrix_test.png",
        show=False,
    )

    print("Todas las visualizaciones guardadas en: {}".format(SAVE_DIR))

    # ============================================================
    # RESUMEN FINAL
    # ============================================================
    print("\n" + "=" * 70)
    print("📋 RESUMEN FINAL")
    print("=" * 70)
    print(f"\n🎯 Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"📈 F1-Score: {test_metrics['f1']:.4f}")
    print(f"🎲 Entropía media: {test_metrics['mean_entropy']:.4f}")
    print(f"🧠 Epistémica: {test_metrics['mean_epistemic']:.4f}")
    print(f"🎲 Aleatoria: {test_metrics['mean_aleatoric']:.4f}")
    print(f"📊 ECE: {test_metrics['ece']:.4f}")
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
    print("=" * 70)


if __name__ == "__main__":
    main()
