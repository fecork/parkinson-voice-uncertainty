"""
CNN Utilities Module
=====================
Utilidades para entrenar CNN reutilizando el pipeline existente:
- Split speaker-independent
- SpecAugment como transform
- Class weights
"""

from typing import Dict, List, Optional
from collections import defaultdict, Counter
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchaudio.transforms as T


# ============================================================
# SPEC AUGMENT (Transform para aplicar on-the-fly)
# ============================================================


class SpecAugment:
    """
    SpecAugment para aplicar como transform on-the-fly.
    Solo se aplica durante entrenamiento.
    """

    def __init__(
        self, freq_mask_param: int = 8, time_mask_param: int = 6, prob: float = 0.5
    ):
        """
        Args:
            freq_mask_param: Máximo número de bins de frecuencia a enmascarar
            time_mask_param: Máximo número de frames de tiempo a enmascarar
            prob: Probabilidad de aplicar cada tipo de máscara
        """
        self.freq_masking = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_masking = T.TimeMasking(time_mask_param=time_mask_param)
        self.prob = prob

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Aplica SpecAugment.

        Args:
            spec: Espectrograma [1, freq, time] o [freq, time]

        Returns:
            Espectrograma aumentado
        """
        # Asegurar que tenga dimensión de canal
        if spec.dim() == 2:
            spec = spec.unsqueeze(0)  # [1, freq, time]

        # Aplicar máscaras con probabilidad
        if random.random() < self.prob:
            spec = self.freq_masking(spec)

        if random.random() < self.prob:
            spec = self.time_masking(spec)

        return spec


# ============================================================
# DATASET CON TRANSFORMS (extiende VowelSegmentsDataset)
# ============================================================


class VowelSegmentsWithTransforms(Dataset):
    """
    Wrapper sobre VowelSegmentsDataset que aplica transforms on-the-fly.
    Reutiliza los espectrogramas ya calculados.
    """

    def __init__(
        self,
        base_dataset,
        transform: Optional[callable] = None,
        indices: Optional[List[int]] = None,
    ):
        """
        Args:
            base_dataset: Dataset base (VowelSegmentsDataset o similar)
            transform: Transform a aplicar (ej: SpecAugment)
            indices: Índices del subset a usar (None = todos)
        """
        self.base_dataset = base_dataset
        self.transform = transform
        self.indices = (
            indices if indices is not None else list(range(len(base_dataset)))
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Obtener item del dataset base
        real_idx = self.indices[idx]
        item = self.base_dataset[real_idx]

        # Aplicar transform si existe
        X = item["X"]  # Ya es (1, 65, 41)

        if self.transform is not None:
            X = self.transform(X)

        return {
            "spectrogram": X,
            "label": item["y_task"],
            "speaker": item["meta"].subject_id,
            "file_id": real_idx,  # Usar índice como file_id
            "meta": item["meta"],
        }


# ============================================================
# SPLIT SPEAKER-INDEPENDENT
# ============================================================


def split_by_speaker(
    metas: List,
    train_ratio: float = 0.6,
    val_ratio: float = 0.15,
    test_ratio: float = 0.25,
    seed: int = 42,
) -> Dict[str, List[int]]:
    """
    Hace split speaker-independent usando los metadatos.

    Args:
        metas: Lista de SampleMeta del dataset
        train_ratio: Proporción para entrenamiento
        val_ratio: Proporción para validación
        test_ratio: Proporción para test
        seed: Semilla para reproducibilidad

    Returns:
        Dict con 'train', 'val', 'test' conteniendo listas de índices
    """
    random.seed(seed)
    np.random.seed(seed)

    # Agrupar índices por speaker
    speaker_to_indices = defaultdict(list)
    speaker_labels = {}

    for idx, meta in enumerate(metas):
        speaker_id = meta.subject_id
        speaker_to_indices[speaker_id].append(idx)

        # Determinar label (asumiendo mismo label por speaker)
        # 0 = HC (h, l, n), 1 = PD (lhl o según mapeo)
        condition = meta.condition
        if condition in ["l", "n"]:
            label = 0  # HC
        elif condition in ["h", "lhl"]:
            label = 1  # PD
        else:
            label = 0  # Default HC

        speaker_labels[speaker_id] = label

    # Separar speakers por clase
    hc_speakers = [s for s, l in speaker_labels.items() if l == 0]
    pd_speakers = [s for s, l in speaker_labels.items() if l == 1]

    print(f"\n📊 Split speaker-independent:")
    print(f"  - HC speakers: {len(hc_speakers)}")
    print(f"  - PD speakers: {len(pd_speakers)}")

    # Split HC speakers
    if len(hc_speakers) > 2:
        hc_train, hc_temp = train_test_split(
            hc_speakers, train_size=train_ratio, random_state=seed
        )

        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        hc_val, hc_test = train_test_split(
            hc_temp, train_size=val_ratio_adjusted, random_state=seed
        )
    else:
        # Muy pocos speakers
        print("⚠️  Pocos speakers HC, usando split simple")
        hc_train = hc_speakers
        hc_val = hc_speakers[: max(1, len(hc_speakers) // 2)]
        hc_test = hc_speakers

    # Split PD speakers
    if len(pd_speakers) > 2:
        pd_train, pd_temp = train_test_split(
            pd_speakers, train_size=train_ratio, random_state=seed
        )

        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        pd_val, pd_test = train_test_split(
            pd_temp, train_size=val_ratio_adjusted, random_state=seed
        )
    else:
        # Muy pocos speakers PD (típico en este dataset)
        print("⚠️  Pocos speakers PD - usando todos en todos los splits")
        msg = "    (ADVERTENCIA: esto causa data leakage pero es necesario"
        print(f"{msg} con 1 speaker)")
        pd_train = pd_speakers
        pd_val = pd_speakers
        pd_test = pd_speakers

    # Construir listas de índices
    def get_indices_for_speakers(speakers):
        indices = []
        for speaker in speakers:
            indices.extend(speaker_to_indices[speaker])
        return indices

    train_indices = get_indices_for_speakers(hc_train + pd_train)
    val_indices = get_indices_for_speakers(hc_val + pd_val)
    test_indices = get_indices_for_speakers(hc_test + pd_test)

    # Estadísticas
    def count_labels(indices):
        labels = [speaker_labels[metas[i].subject_id] for i in indices]
        return Counter(labels)

    train_dist = dict(count_labels(train_indices))
    val_dist = dict(count_labels(val_indices))
    test_dist = dict(count_labels(test_indices))

    print(f"\n  Train: {len(train_indices)} segmentos - {train_dist}")
    print(f"  Val:   {len(val_indices)} segmentos - {val_dist}")
    print(f"  Test:  {len(test_indices)} segmentos - {test_dist}")

    return {"train": train_indices, "val": val_indices, "test": test_indices}


# ============================================================
# CREACIÓN DE DATALOADERS
# ============================================================


def create_dataloaders_from_existing(
    base_dataset,
    split_indices: Dict[str, List[int]],
    batch_size: int = 32,
    spec_augment_params: Optional[Dict] = None,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """
    Crea DataLoaders a partir del dataset existente.

    Args:
        base_dataset: Dataset base (VowelSegmentsDataset)
        split_indices: Output de split_by_speaker
        batch_size: Tamaño de batch
        spec_augment_params: Parámetros para SpecAugment (solo train)
        num_workers: Número de workers

    Returns:
        Dict con 'train', 'val', 'test' DataLoaders
    """
    if spec_augment_params is None:
        spec_augment_params = {"freq_mask_param": 8, "time_mask_param": 6, "prob": 0.5}

    # Crear SpecAugment
    spec_augment = SpecAugment(**spec_augment_params)

    # Crear datasets con/sin augmentation
    train_dataset = VowelSegmentsWithTransforms(
        base_dataset, transform=spec_augment, indices=split_indices["train"]
    )

    val_dataset = VowelSegmentsWithTransforms(
        base_dataset,
        transform=None,  # Sin augmentation
        indices=split_indices["val"],
    )

    test_dataset = VowelSegmentsWithTransforms(
        base_dataset, transform=None, indices=split_indices["test"]
    )

    # Crear DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"\n✅ DataLoaders creados:")
    print(f"  - Train: {len(train_dataset)} segmentos")
    print(f"  - Val:   {len(val_dataset)} segmentos")
    print(f"  - Test:  {len(test_dataset)} segmentos")
    print(f"  - Batch size: {batch_size}")
    print(f"  - SpecAugment en train: Sí")

    return {"train": train_loader, "val": val_loader, "test": test_loader}


# ============================================================
# CLASS WEIGHTS
# ============================================================


def compute_class_weights_from_dataset(
    dataset, indices: Optional[List[int]] = None
) -> torch.Tensor:
    """
    Calcula class weights desde un dataset.

    Args:
        dataset: Dataset PyTorch
        indices: Índices a considerar (None = todos)

    Returns:
        Tensor [2] con pesos por clase
    """
    if indices is None:
        indices = list(range(len(dataset)))

    # Contar labels
    labels = []
    for idx in indices:
        item = dataset[idx]
        if isinstance(item, dict):
            label = item.get("y_task", item.get("label"))
        else:
            label = item[1]  # Asumiendo (X, y) tuple

        if torch.is_tensor(label):
            label = label.item()
        labels.append(label)

    counts = Counter(labels)
    total = len(labels)

    # Weight inversamente proporcional
    weights = torch.tensor(
        [
            total / (2 * counts[0]) if 0 in counts else 1.0,
            total / (2 * counts[1]) if 1 in counts else 1.0,
        ],
        dtype=torch.float32,
    )

    print(f"\n⚖️  Class weights calculados:")
    print(f"   Distribución: HC={counts.get(0, 0)}, PD={counts.get(1, 0)}")
    print(f"   Weights: {weights.numpy()}")

    return weights
