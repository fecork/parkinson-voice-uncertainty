"""
Sequence Dataset Module
========================
Funciones para crear secuencias de espectrogramas para modelos LSTM.

Time-CNN-BiLSTM requiere secuencias de n espectrogramas consecutivos
del mismo archivo de audio, con zero-padding y masking para secuencias cortas.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from pathlib import Path
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


# ============================================================
# SEQUENCE GROUPING
# ============================================================


def normalize_sequence(sequence: np.ndarray, length: int) -> np.ndarray:
    """
    Normaliza una secuencia completa manteniendo coherencia temporal.

    CRÍTICO para LSTM: Normaliza usando estadísticas de TODA la secuencia
    válida (sin padding), no frame por frame. Esto preserva las relaciones
    temporales entre frames consecutivos.

    Args:
        sequence: Array (T, C, H, W) con padding
        length: Número de frames válidos (sin padding)

    Returns:
        Secuencia normalizada manteniendo estructura temporal y padding
        en cero
    """
    # Extract valid frames only for statistics
    valid_frames = sequence[:length]

    # Compute statistics over ALL valid frames together
    mean = valid_frames.mean()
    std = valid_frames.std()

    # Normalize valid frames
    sequence_normalized = np.copy(sequence)
    sequence_normalized[:length] = (valid_frames - mean) / (std + 1e-8)

    # Keep padding as zeros
    sequence_normalized[length:] = 0.0

    return sequence_normalized


def group_spectrograms_to_sequences(
    dataset: List[Dict],
    n_frames: int = 7,
    min_frames: int = 3,
    normalize: bool = False,
) -> Tuple[List[np.ndarray], List[int], List[Dict]]:
    """
    Agrupa espectrogramas consecutivos del mismo audio en secuencias.

    Según Ibarra et al. (2023), los espectrogramas ya están normalizados
    individualmente con z-score. Para Time-CNN-LSTM, simplemente agrupamos
    espectrogramas consecutivos del mismo audio con zero-padding.

    Args:
        dataset: Lista de dicts con 'spectrogram' y 'metadata'
        n_frames: Número objetivo de frames por secuencia
        min_frames: Número mínimo de frames (descartar si menos)
        normalize: Si True, re-normaliza por secuencia (False por defecto,
                   ya normalizado individualmente según paper)

    Returns:
        sequences: Lista de arrays (n_frames, 1, H, W) con zero-padding
        lengths: Lista de longitudes reales (para masking)
        metadata: Lista de metadata del primer frame de cada secuencia
    """
    # Agrupar por archivo de audio (subject_id + filename)
    audio_groups = defaultdict(list)

    for i, sample in enumerate(dataset):
        meta = sample["metadata"]
        # Clave única por archivo de audio
        audio_key = f"{meta.subject_id}_{meta.filename}"
        audio_groups[audio_key].append((i, sample))

    sequences = []
    lengths = []
    sequence_metas = []

    print(f"\n[INFO] Agrupando espectrogramas en secuencias de {n_frames} frames...")
    print(f"   Archivos unicos encontrados: {len(audio_groups)}")
    if normalize:
        print(
            f"   [CONFIG] Normalizacion: POR SECUENCIA (mantiene continuidad temporal)"
        )
    else:
        print(
            f"   [CONFIG] Normalizacion: YA APLICADA (z-score por espectrograma individual)"
        )

    discarded = 0
    for audio_key, samples in audio_groups.items():
        # Ordenar por segment_id para mantener orden temporal
        samples = sorted(samples, key=lambda x: x[1]["metadata"].segment_id)

        # Extraer espectrogramas
        specs = [s[1]["spectrogram"] for s in samples]
        num_specs = len(specs)

        # Descartar si muy pocos frames
        if num_specs < min_frames:
            discarded += 1
            continue

        # Crear secuencia con padding
        sequence = np.zeros((n_frames, 1, 65, 41), dtype=np.float32)
        actual_length = min(num_specs, n_frames)

        # Llenar con espectrogramas reales
        for i in range(actual_length):
            # specs[i] ya tiene shape (65, 41) o (1, 65, 41)
            spec = specs[i]
            if spec.ndim == 2:
                spec = np.expand_dims(spec, axis=0)  # (1, H, W)
            sequence[i] = spec

        # NUEVO: Normalizar por secuencia completa
        if normalize:
            sequence = normalize_sequence(sequence, actual_length)

        sequences.append(sequence)
        lengths.append(actual_length)

        # Metadata del primer frame
        sequence_metas.append(samples[0][1]["metadata"])

    print(f"   [OK] {len(sequences)} secuencias creadas")
    if discarded > 0:
        print(f"   [WARN] {discarded} archivos descartados (< {min_frames} frames)")

    return sequences, lengths, sequence_metas


def save_sequence_cache(
    sequences: List[np.ndarray],
    lengths: List[int],
    metadata: List,
    cache_path: str,
) -> None:
    """
    Guarda secuencias en archivo cache.

    Args:
        sequences: Lista de secuencias
        lengths: Lista de longitudes
        metadata: Lista de metadata
        cache_path: Path al archivo cache
    """
    cache_data = {
        "sequences": sequences,
        "lengths": lengths,
        "metadata": metadata,
    }

    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)

    with open(cache_path, "wb") as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = Path(cache_path).stat().st_size / (1024 * 1024)
    print(f"\n💾 Cache guardado: {cache_path}")
    print(f"   Tamaño: {size_mb:.1f} MB")


def load_sequence_cache(cache_path: str) -> Optional[Dict]:
    """
    Carga secuencias desde archivo cache.

    Args:
        cache_path: Path al archivo cache

    Returns:
        Dict con 'sequences', 'lengths', 'metadata' o None si no existe
    """
    if not Path(cache_path).exists():
        return None

    try:
        with open(cache_path, "rb") as f:
            cache_data = pickle.load(f)

        print(f"✅ Cache cargado: {cache_path}")
        print(f"   Secuencias: {len(cache_data['sequences'])}")

        return cache_data
    except Exception as e:
        print(f"⚠️  Error cargando cache: {e}")
        return None


# ============================================================
# DOMAIN MAPPING
# ============================================================


def create_domain_mapping_from_subjects(metadata_list: List) -> Dict[str, int]:
    """
    Crea mapeo determinístico de subject_id a domain_id (0-3) para Domain Adaptation.

    Estrategia basada en paper Ibarra et al. (2023):
    - 4 dominios fijos (GITA, Neurovoz, German, Czech)
    - Healthy subjects: distribuir en dominios 0, 1, 2
    - Parkinson subjects: asignar a dominio 3

    Args:
        metadata_list: Lista de metadata con subject_id

    Returns:
        Dict mapeando subject_id -> domain_id (0-3)
    """
    # Extraer subject_ids únicos
    subject_ids = set()
    for meta in metadata_list:
        subject_ids.add(meta.subject_id)

    subject_ids = sorted(list(subject_ids))  # Ordenar para determinismo

    # Estrategia de mapeo:
    # - Healthy subjects (no 1580): distribuir en dominios 0, 1, 2
    # - Parkinson subject (1580): asignar a dominio 3
    domain_mapping = {}

    for i, subject_id in enumerate(subject_ids):
        if subject_id == 1580:  # Parkinson subject
            domain_mapping[subject_id] = 3
        else:  # Healthy subjects
            domain_mapping[subject_id] = i % 3  # Distribuir en dominios 0, 1, 2

    return domain_mapping


# ============================================================
# PYTORCH DATASET
# ============================================================


class SequenceLSTMDataset(Dataset):
    """
    PyTorch Dataset para secuencias de espectrogramas con masking.

    Usado por Time-CNN-BiLSTM para cargar secuencias (B, T, 1, H, W)
    con sus longitudes reales para masking en LSTM.
    """

    def __init__(
        self,
        sequences: List[np.ndarray],
        lengths: List[int],
        labels: List[int],
        domain_labels: List[int],
        metadata: List,
    ):
        """
        Args:
            sequences: Lista de arrays (n_frames, 1, H, W)
            lengths: Lista de longitudes reales
            labels: Lista de etiquetas de tarea (0=HC, 1=PD)
            domain_labels: Lista de etiquetas de dominio
            metadata: Lista de metadata
        """
        assert len(sequences) == len(lengths) == len(labels) == len(metadata), (
            "Length mismatch entre sequences, lengths, labels, metadata"
        )

        self.sequences = sequences
        self.lengths = lengths
        self.labels = labels
        self.domain_labels = domain_labels
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        """
        Retorna una secuencia con su metadata.

        Returns:
            Dict con:
                - X: Tensor (T, 1, H, W)
                - length: int (longitud real)
                - y_task: int (etiqueta PD)
                - y_domain: int (etiqueta dominio)
                - meta: metadata original
        """
        return {
            "X": torch.from_numpy(self.sequences[idx]).float(),
            "length": self.lengths[idx],
            "y_task": self.labels[idx],
            "y_domain": self.domain_labels[idx],
            "meta": self.metadata[idx],
        }


def create_sequence_dataset_from_cache(
    cache_path: str,
    label_value: int,
) -> Optional[SequenceLSTMDataset]:
    """
    Crea un SequenceLSTMDataset desde cache.

    Args:
        cache_path: Path al archivo cache
        label_value: Valor de etiqueta (0=HC, 1=PD)

    Returns:
        SequenceLSTMDataset o None si no existe cache
    """
    cache_data = load_sequence_cache(cache_path)

    if cache_data is None:
        return None

    # Crear labels
    n_samples = len(cache_data["sequences"])
    labels = [label_value] * n_samples

    # Crear domain labels (basado en subject_id para 4 dominios fijos)
    domain_mapping = create_domain_mapping_from_subjects(cache_data["metadata"])
    domain_labels = []

    for meta in cache_data["metadata"]:
        subject_id = meta.subject_id
        domain_labels.append(domain_mapping[subject_id])

    return SequenceLSTMDataset(
        sequences=cache_data["sequences"],
        lengths=cache_data["lengths"],
        labels=labels,
        domain_labels=domain_labels,
        metadata=cache_data["metadata"],
    )


# ============================================================
# STATISTICS
# ============================================================


def print_sequence_stats(
    sequences: List[np.ndarray],
    lengths: List[int],
    label: str = "Dataset",
) -> None:
    """
    Imprime estadísticas de secuencias.

    Args:
        sequences: Lista de secuencias
        lengths: Lista de longitudes
        label: Etiqueta para el print
    """
    print(f"\n📊 ESTADÍSTICAS DE SECUENCIAS - {label}")
    print(f"   Total secuencias: {len(sequences)}")

    if len(sequences) > 0:
        print(f"   Shape por secuencia: {sequences[0].shape}")

        # Estadísticas de longitudes
        lengths_array = np.array(lengths)
        print("\n   Longitudes (frames reales):")
        print(f"      • Mean: {lengths_array.mean():.1f}")
        print(f"      • Std: {lengths_array.std():.1f}")
        print(f"      • Min: {lengths_array.min()}")
        print(f"      • Max: {lengths_array.max()}")

        # Distribución de longitudes
        unique, counts = np.unique(lengths_array, return_counts=True)
        print("\n   Distribución:")
        for length, count in zip(unique, counts):
            pct = count / len(lengths) * 100
            print(f"      • {length} frames: {count} ({pct:.1f}%)")
