"""
CNN Inference Module
=====================
Inferencia con MC Dropout, agregación por archivo/paciente,
y análisis de incertidumbre.
"""

from typing import Dict, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model import enable_dropout


# ============================================================
# MC DROPOUT INFERENCE
# ============================================================


@torch.no_grad()
def mc_dropout_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_samples: int = 30,
    verbose: bool = True,
) -> Dict:
    """
    Inferencia con MC Dropout para todo el dataset.

    Args:
        model: Modelo PyTorch
        loader: DataLoader
        device: Device para cómputo
        n_samples: Número de forward passes estocásticos
        verbose: Si True, imprime progreso

    Returns:
        Dict con:
            - predictions: Predicciones finales (N,)
            - probabilities_mean: Probabilidades promedio (N, n_classes)
            - probabilities_std: Desv. estándar (N, n_classes)
            - entropy: Entropía predictiva (N,)
            - labels: Etiquetas verdaderas (N,)
            - speakers: IDs de speakers (N,)
            - file_ids: IDs de archivos (N,)
    """
    if verbose:
        print(f"\n🔄 Inferencia MC Dropout con {n_samples} muestras...")

    model.train()
    enable_dropout(model)

    all_probs_list = []  # Lista de listas para cada batch
    all_labels = []
    all_speakers = []
    all_file_ids = []

    for batch_idx, batch in enumerate(loader):
        specs = batch["spectrogram"].to(device)
        labels = batch["label"]
        speakers = batch["speaker"]
        file_ids = batch["file_id"]

        # MC Dropout: múltiples forward passes
        batch_probs = []
        for _ in range(n_samples):
            logits = model(specs)
            probs = F.softmax(logits, dim=1)
            batch_probs.append(probs.cpu())

        # Stack: (n_samples, B, n_classes)
        batch_probs = torch.stack(batch_probs, dim=0)
        all_probs_list.append(batch_probs)

        all_labels.extend(labels.numpy())
        all_speakers.extend(speakers)
        all_file_ids.extend(file_ids.numpy())

        if verbose and (batch_idx + 1) % 10 == 0:
            print(f"  Procesados {(batch_idx + 1) * len(specs)} segmentos...")

    # Concatenar todos los batches: (n_samples, N, n_classes)
    all_probs = torch.cat(all_probs_list, dim=1)

    # Estadísticas
    probs_mean = all_probs.mean(dim=0)  # (N, n_classes)
    probs_std = all_probs.std(dim=0)  # (N, n_classes)

    # Predicciones
    predictions = probs_mean.argmax(dim=1).numpy()

    # Entropía predictiva
    entropy = -torch.sum(probs_mean * torch.log(probs_mean + 1e-10), dim=1).numpy()

    # Varianza total (promedio sobre clases)
    variance = probs_std.mean(dim=1).numpy()

    model.eval()

    if verbose:
        print(f"✅ Inferencia completada: {len(predictions)} segmentos")

    return {
        "predictions": predictions,
        "probabilities_mean": probs_mean.numpy(),
        "probabilities_std": probs_std.numpy(),
        "entropy": entropy,
        "variance": variance,
        "labels": np.array(all_labels),
        "speakers": all_speakers,
        "file_ids": np.array(all_file_ids),
    }


# ============================================================
# AGREGACIÓN POR ARCHIVO/PACIENTE
# ============================================================


def aggregate_by_file(mc_results: Dict, aggregation_method: str = "mean") -> Dict:
    """
    Agrega predicciones por archivo.

    Args:
        mc_results: Output de mc_dropout_inference
        aggregation_method: 'mean', 'median', 'vote'

    Returns:
        Dict con resultados agregados por archivo:
            - file_predictions: (n_files,)
            - file_probabilities: (n_files, n_classes)
            - file_uncertainty: (n_files,)
            - file_labels: (n_files,)
            - file_ids: (n_files,)
    """
    # Agrupar por file_id
    file_groups = defaultdict(list)

    for i, file_id in enumerate(mc_results["file_ids"]):
        file_groups[file_id].append(i)

    n_classes = mc_results["probabilities_mean"].shape[1]

    file_predictions = []
    file_probabilities = []
    file_uncertainty = []
    file_labels = []
    file_ids = []

    for file_id in sorted(file_groups.keys()):
        indices = file_groups[file_id]

        # Obtener segmentos del archivo
        probs = mc_results["probabilities_mean"][indices]
        entropies = mc_results["entropy"][indices]
        label = mc_results["labels"][indices[0]]

        # Agregación de probabilidades
        if aggregation_method == "mean":
            agg_probs = probs.mean(axis=0)
        elif aggregation_method == "median":
            agg_probs = np.median(probs, axis=0)
        elif aggregation_method == "vote":
            # Majority voting
            preds = probs.argmax(axis=1)
            counts = np.bincount(preds, minlength=n_classes)
            agg_probs = counts / len(preds)
        else:
            raise ValueError(f"Método desconocido: {aggregation_method}")

        # Predicción final
        pred = agg_probs.argmax()

        # Incertidumbre agregada (promedio de entropías)
        agg_uncertainty = entropies.mean()

        file_predictions.append(pred)
        file_probabilities.append(agg_probs)
        file_uncertainty.append(agg_uncertainty)
        file_labels.append(label)
        file_ids.append(file_id)

    return {
        "file_predictions": np.array(file_predictions),
        "file_probabilities": np.array(file_probabilities),
        "file_uncertainty": np.array(file_uncertainty),
        "file_labels": np.array(file_labels),
        "file_ids": np.array(file_ids),
        "n_segments_per_file": {
            fid: len(file_groups[fid]) for fid in sorted(file_groups.keys())
        },
    }


def aggregate_by_patient(
    mc_results: Dict,
    file_to_patient: Optional[Dict] = None,
    aggregation_method: str = "mean",
) -> Dict:
    """
    Agrega predicciones por paciente.

    Args:
        mc_results: Output de mc_dropout_inference
        file_to_patient: Mapping file_id -> patient_id
        aggregation_method: 'mean', 'median', 'vote'

    Returns:
        Dict con resultados agregados por paciente
    """
    # Si no hay mapping, usar speaker como patient_id
    if file_to_patient is None:
        file_to_patient = {}
        for i, file_id in enumerate(mc_results["file_ids"]):
            speaker = mc_results["speakers"][i]
            file_to_patient[file_id] = speaker

    # Agrupar por paciente
    patient_groups = defaultdict(list)

    for i, file_id in enumerate(mc_results["file_ids"]):
        patient_id = file_to_patient.get(file_id, "unknown")
        patient_groups[patient_id].append(i)

    n_classes = mc_results["probabilities_mean"].shape[1]

    patient_predictions = []
    patient_probabilities = []
    patient_uncertainty = []
    patient_labels = []
    patient_ids = []

    for patient_id in sorted(patient_groups.keys()):
        indices = patient_groups[patient_id]

        # Obtener todos los segmentos del paciente
        probs = mc_results["probabilities_mean"][indices]
        entropies = mc_results["entropy"][indices]
        label = mc_results["labels"][indices[0]]

        # Agregación
        if aggregation_method == "mean":
            agg_probs = probs.mean(axis=0)
        elif aggregation_method == "median":
            agg_probs = np.median(probs, axis=0)
        elif aggregation_method == "vote":
            preds = probs.argmax(axis=1)
            counts = np.bincount(preds, minlength=n_classes)
            agg_probs = counts / len(preds)
        else:
            raise ValueError(f"Método desconocido: {aggregation_method}")

        pred = agg_probs.argmax()
        agg_uncertainty = entropies.mean()

        patient_predictions.append(pred)
        patient_probabilities.append(agg_probs)
        patient_uncertainty.append(agg_uncertainty)
        patient_labels.append(label)
        patient_ids.append(patient_id)

    return {
        "patient_predictions": np.array(patient_predictions),
        "patient_probabilities": np.array(patient_probabilities),
        "patient_uncertainty": np.array(patient_uncertainty),
        "patient_labels": np.array(patient_labels),
        "patient_ids": patient_ids,
        "n_segments_per_patient": {
            pid: len(patient_groups[pid]) for pid in sorted(patient_groups.keys())
        },
    }


# ============================================================
# ANÁLISIS DE INCERTIDUMBRE
# ============================================================


def analyze_uncertainty(
    mc_results: Dict,
    aggregated_results: Optional[Dict] = None,
    threshold_percentile: float = 75,
) -> Dict:
    """
    Analiza distribución de incertidumbre.

    Args:
        mc_results: Output de mc_dropout_inference (nivel segmento)
        aggregated_results: Output de aggregate_by_file/patient (opcional)
        threshold_percentile: Percentil para definir "alta incertidumbre"

    Returns:
        Dict con análisis de incertidumbre
    """
    entropy = mc_results["entropy"]
    variance = mc_results["variance"]
    predictions = mc_results["predictions"]
    labels = mc_results["labels"]

    # Correctness
    correct = predictions == labels

    # Estadísticas de incertidumbre
    analysis = {
        "entropy": {
            "mean": float(entropy.mean()),
            "std": float(entropy.std()),
            "min": float(entropy.min()),
            "max": float(entropy.max()),
            "median": float(np.median(entropy)),
            "percentile_75": float(np.percentile(entropy, 75)),
            "percentile_90": float(np.percentile(entropy, 90)),
        },
        "variance": {
            "mean": float(variance.mean()),
            "std": float(variance.std()),
            "min": float(variance.min()),
            "max": float(variance.max()),
            "median": float(np.median(variance)),
        },
    }

    # Threshold de alta incertidumbre
    high_uncertainty_threshold = np.percentile(entropy, threshold_percentile)
    high_uncertainty_mask = entropy > high_uncertainty_threshold

    analysis["high_uncertainty"] = {
        "threshold": float(high_uncertainty_threshold),
        "n_samples": int(high_uncertainty_mask.sum()),
        "percentage": float(high_uncertainty_mask.mean() * 100),
    }

    # Relación incertidumbre-correctness
    analysis["uncertainty_vs_correctness"] = {
        "correct_mean_entropy": float(entropy[correct].mean()),
        "incorrect_mean_entropy": float(entropy[~correct].mean()),
        "correct_mean_variance": float(variance[correct].mean()),
        "incorrect_mean_variance": float(variance[~correct].mean()),
    }

    # Por clase
    for class_id in [0, 1]:
        class_mask = labels == class_id
        class_name = "HC" if class_id == 0 else "PD"

        analysis[f"class_{class_name}"] = {
            "mean_entropy": float(entropy[class_mask].mean()),
            "mean_variance": float(variance[class_mask].mean()),
            "n_samples": int(class_mask.sum()),
        }

    # Si hay resultados agregados, analizar también
    if aggregated_results is not None:
        if "file_uncertainty" in aggregated_results:
            file_unc = aggregated_results["file_uncertainty"]
            analysis["file_level"] = {
                "mean_uncertainty": float(file_unc.mean()),
                "std_uncertainty": float(file_unc.std()),
                "median_uncertainty": float(np.median(file_unc)),
            }

        if "patient_uncertainty" in aggregated_results:
            patient_unc = aggregated_results["patient_uncertainty"]
            analysis["patient_level"] = {
                "mean_uncertainty": float(patient_unc.mean()),
                "std_uncertainty": float(patient_unc.std()),
                "median_uncertainty": float(np.median(patient_unc)),
            }

    return analysis


def find_interesting_cases(mc_results: Dict, n_per_category: int = 5) -> Dict:
    """
    Encuentra casos interesantes para análisis/visualización.

    Categorías:
    - Acierto con alta confianza (baja incertidumbre)
    - Error con baja confianza (alta incertidumbre) - esperado
    - Error con alta confianza (baja incertidumbre) - inesperado!

    Args:
        mc_results: Output de mc_dropout_inference
        n_per_category: Número de casos por categoría

    Returns:
        Dict con índices de casos interesantes
    """
    predictions = mc_results["predictions"]
    labels = mc_results["labels"]
    entropy = mc_results["entropy"]

    correct = predictions == labels

    # Casos interesantes
    cases = {
        "correct_confident": [],  # Acierto + baja incertidumbre
        "incorrect_uncertain": [],  # Error + alta incertidumbre
        "incorrect_confident": [],  # Error + baja incertidumbre (PROBLEMA!)
    }

    # Aciertos con alta confianza (baja entropía)
    correct_idx = np.where(correct)[0]
    if len(correct_idx) > 0:
        sorted_idx = correct_idx[np.argsort(entropy[correct_idx])]
        cases["correct_confident"] = sorted_idx[:n_per_category].tolist()

    # Errores
    incorrect_idx = np.where(~correct)[0]
    if len(incorrect_idx) > 0:
        # Errores con alta incertidumbre (comportamiento esperado)
        sorted_uncertain = incorrect_idx[np.argsort(-entropy[incorrect_idx])]
        cases["incorrect_uncertain"] = sorted_uncertain[:n_per_category].tolist()

        # Errores con baja incertidumbre (comportamiento problemático)
        sorted_confident = incorrect_idx[np.argsort(entropy[incorrect_idx])]
        cases["incorrect_confident"] = sorted_confident[:n_per_category].tolist()

    return cases


# ============================================================
# REPORTE
# ============================================================


def print_inference_report(
    mc_results: Dict,
    file_results: Optional[Dict] = None,
    patient_results: Optional[Dict] = None,
):
    """
    Imprime reporte de inferencia.

    Args:
        mc_results: Resultados a nivel segmento
        file_results: Resultados a nivel archivo (opcional)
        patient_results: Resultados a nivel paciente (opcional)
    """
    from sklearn.metrics import accuracy_score, f1_score

    print("\n" + "=" * 70)
    print("REPORTE DE INFERENCIA CON MC DROPOUT")
    print("=" * 70)

    # Nivel segmento
    acc_seg = accuracy_score(mc_results["labels"], mc_results["predictions"])
    f1_seg = f1_score(mc_results["labels"], mc_results["predictions"])

    print(f"\n📊 NIVEL SEGMENTO ({len(mc_results['predictions'])} segmentos):")
    print(f"  Accuracy: {acc_seg:.4f}")
    print(f"  F1-Score: {f1_seg:.4f}")
    print(f"  Entropía promedio: {mc_results['entropy'].mean():.4f}")
    print(f"  Varianza promedio: {mc_results['variance'].mean():.4f}")

    # Nivel archivo
    if file_results is not None:
        acc_file = accuracy_score(
            file_results["file_labels"], file_results["file_predictions"]
        )
        f1_file = f1_score(
            file_results["file_labels"], file_results["file_predictions"]
        )

        print(f"\n📁 NIVEL ARCHIVO ({len(file_results['file_predictions'])} archivos):")
        print(f"  Accuracy: {acc_file:.4f}")
        print(f"  F1-Score: {f1_file:.4f}")
        print(
            f"  Incertidumbre promedio: {file_results['file_uncertainty'].mean():.4f}"
        )

    # Nivel paciente
    if patient_results is not None:
        acc_patient = accuracy_score(
            patient_results["patient_labels"], patient_results["patient_predictions"]
        )
        f1_patient = f1_score(
            patient_results["patient_labels"], patient_results["patient_predictions"]
        )

        print(
            f"\n👤 NIVEL PACIENTE ({len(patient_results['patient_predictions'])} pacientes):"
        )
        print(f"  Accuracy: {acc_patient:.4f}")
        print(f"  F1-Score: {f1_patient:.4f}")
        print(
            f"  Incertidumbre promedio: {patient_results['patient_uncertainty'].mean():.4f}"
        )

    print("=" * 70 + "\n")
