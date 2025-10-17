"""
Dataset Pipeline Module
========================
Pipeline completo para crear datasets PyTorch desde archivos de audio.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from collections import Counter
import numpy as np
import torch

from . import preprocessing


# ============================================================
# DATA STRUCTURES
# ============================================================


@dataclass(frozen=True)
class SampleMeta:
    """Lightweight metadata holder for each audio segment."""

    subject_id: str
    vowel_type: str
    condition: str
    filename: str
    segment_id: int
    sr: int


# ============================================================
# HELPER FUNCTIONS
# ============================================================


def _safe_len(x: Optional[Sequence]) -> int:
    """Return 0 when x is None."""
    return len(x) if x is not None else 0


def _print_progress(i: int, total: int, path_name: str, every: int) -> None:
    """Print compact progress every N files."""
    if i % max(1, every) == 0:
        print(f"  {i + 1}/{total}: {path_name}")


def parse_filename(file_stem: str) -> Tuple[str, str, str]:
    """
    Parse a filename stem into (subject_id, vowel_type, condition).

    Rules:
    - Split by '-'
    - Missing pieces get sensible defaults.
    """
    parts = file_stem.split("-")
    subject_id = parts[0] if len(parts) > 0 and parts[0] else "unknown"
    vowel_type = parts[1] if len(parts) > 1 and parts[1] else "a"
    condition = parts[2] if len(parts) > 2 and parts[2] else "unknown"
    return subject_id, vowel_type, condition


def build_domain_index(vowels: Iterable[str]) -> Dict[str, int]:
    """
    Create a deterministic domain index per vowel (0..K-1) without using hash().
    Ensures reproducibility across runs and machines.
    """
    uniq = sorted(set(vowels))
    return {v: idx for idx, v in enumerate(uniq)}


def map_condition_to_task(condition: str) -> int:
    """
    Map condition labels to a binary task (0=Control, 1=Parkinson).
    Adjust here to fit your dataset semantics.
    """
    mapping = {
        "h": 1,  # Parkinson
        "l": 0,  # Control
        "n": 0,  # Control
        "lhl": 1,  # Parkinson
    }
    return mapping.get(condition, 0)


# ============================================================
# DATASET PROCESSING
# ============================================================


def process_dataset(
    audio_files: Sequence,
    preprocess_fn: Optional[Callable] = None,
    max_files: Optional[int] = None,
    progress_every: int = 10,
    default_sr: int = 44100,
) -> List[Dict]:
    """
    Process the dataset using the paper's preprocessing function.

    Args:
        audio_files: Iterable of pathlib.Path-like objects.
        preprocess_fn: Callable that returns (spectrograms, segments) per file.
        max_files: Optional cap on number of files to process.
        progress_every: Print progress every N files.
        default_sr: Sampling rate to attach to metadata (if unknown externally).

    Returns:
        A list of dict samples with spectrograms, segments, and metadata.
    """
    if preprocess_fn is None:
        preprocess_fn = preprocessing.preprocess_audio_paper

    if not audio_files:
        print("❌ 'audio_files' está vacío: no hay nada que procesar.")
        return []

    files_to_process = list(audio_files[:max_files]) if max_files else list(audio_files)
    print(f"🔄 Procesando {len(files_to_process)} archivos...")

    dataset: List[Dict] = []

    for i, file_path in enumerate(files_to_process):
        _print_progress(
            i,
            len(files_to_process),
            getattr(file_path, "name", str(file_path)),
            progress_every,
        )

        subject_id, vowel_type, condition = parse_filename(
            getattr(file_path, "stem", str(file_path))
        )

        # Llamada al preprocesamiento (del paper)
        try:
            spectrograms, segments = preprocess_fn(file_path, vowel_type=vowel_type)
        except Exception as e:
            print(f"⚠️  Error al procesar {file_path}: {e}. Continuando…")
            continue

        if not spectrograms:
            # Nada que agregar de este archivo
            continue

        # Empaquetar muestras
        for j, (spec, seg) in enumerate(zip(spectrograms, segments)):
            dataset.append(
                {
                    "spectrogram": spec,  # numpy array 2D
                    "segment": seg,  # numpy array 1D (si aplica)
                    "metadata": SampleMeta(
                        subject_id=subject_id,
                        vowel_type=vowel_type,
                        condition=condition,
                        filename=getattr(file_path, "name", str(file_path)),
                        segment_id=j,
                        sr=default_sr,
                    ),
                }
            )

    print(f"✅ {len(dataset)} muestras generadas")
    return dataset


def to_pytorch_tensors(
    dataset: List[Dict],
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    List[SampleMeta],
]:
    """
    Convert the processed dataset to PyTorch tensors.

    Returns:
        X        : FloatTensor (N, 1, H, W) spectrograms
        y_task   : LongTensor (N,)     task labels
        y_domain : LongTensor (N,)     domain labels (by vowel)
        metas    : List[SampleMeta]    metadata list
    """
    if not dataset:
        print("❌ Dataset vacío: no hay tensores que crear.")
        return None, None, None, []

    # Extraer metadatos y espectrogramas
    metas: List[SampleMeta] = [sample["metadata"] for sample in dataset]
    vowels = [m.vowel_type for m in metas]
    domain_index = build_domain_index(vowels)

    specs: List[np.ndarray] = []
    y_task: List[int] = []
    y_domain: List[int] = []

    for sample in dataset:
        spec: np.ndarray = sample["spectrogram"]
        if spec.ndim != 2:
            raise ValueError(f"Spectrogram must be 2D, got shape: {spec.shape}")

        # canal = 1 para CNN 2D
        specs.append(np.expand_dims(spec, axis=0))  # (1, H, W)
        y_task.append(map_condition_to_task(sample["metadata"].condition))
        y_domain.append(domain_index[sample["metadata"].vowel_type])

    X = torch.from_numpy(np.stack(specs, axis=0)).float()  # (N, 1, H, W)
    y_task_t = torch.tensor(y_task, dtype=torch.long)  # (N,)
    y_domain_t = torch.tensor(y_domain, dtype=torch.long)  # (N,)

    # Reporte compacto
    print("📊 PyTorch tensors listos:")
    print(f"  - X: {tuple(X.shape)}")
    print(f"  - y_task: {tuple(y_task_t.shape)}  (dist={dict(Counter(y_task))})")
    print(f"  - y_domain: {tuple(y_domain_t.shape)}  (K dominios={len(domain_index)})")

    return X, y_task_t, y_domain_t, metas


# ============================================================
# PYTORCH DATASET
# ============================================================


class VowelSegmentsDataset(torch.utils.data.Dataset):
    """A thin PyTorch Dataset wrapper for training."""

    def __init__(
        self,
        X: torch.Tensor,
        y_task: torch.Tensor,
        y_domain: torch.Tensor,
        metas: List[SampleMeta],
    ):
        assert X is not None and y_task is not None and y_domain is not None, (
            "Tensors must not be None"
        )
        assert len(X) == len(y_task) == len(y_domain) == len(metas), (
            "Length mismatch between tensors and metadata"
        )
        self.X = X
        self.y_task = y_task
        self.y_domain = y_domain
        self.metas = metas

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return {
            "X": self.X[idx],  # (1, H, W)
            "y_task": self.y_task[idx],  # scalar
            "y_domain": self.y_domain[idx],  # scalar
            "meta": self.metas[idx],  # SampleMeta
        }


# ============================================================
# SUMMARY UTILITIES
# ============================================================


def summarize_distribution(dataset: List[Dict]) -> Dict[str, Counter]:
    """
    Compute distributions by vowel and condition from dataset metadata.
    """
    vowels = Counter()
    conditions = Counter()
    for sample in dataset:
        m: SampleMeta = sample["metadata"]
        vowels[m.vowel_type] += 1
        conditions[m.condition] += 1
    return {"vowel": vowels, "condition": conditions}


def print_summary(dist: Dict[str, Counter]) -> None:
    """Pretty-print distributions."""
    print("\n📊 DISTRIBUCIÓN POR VOCAL:")
    for k, v in dist["vowel"].items():
        print(f"  - {k}: {v} muestras")
    print("\n📊 DISTRIBUCIÓN POR CONDICIÓN:")
    for k, v in dist["condition"].items():
        print(f"  - {k}: {v} muestras")


# ============================================================
# FULL PIPELINE
# ============================================================


def build_full_pipeline(
    audio_files: Optional[Sequence],
    preprocess_fn: Optional[Callable] = None,
    max_files: Optional[int] = None,
):
    """
    One-shot pipeline to produce:
      - raw dataset (list of dicts)
      - PyTorch tensors (X, y_task, y_domain)
      - torch Dataset (VowelSegmentsDataset)
      - summary distributions
    Returns robust empty outputs if audio_files is None or empty.
    """
    if preprocess_fn is None:
        preprocess_fn = preprocessing.preprocess_audio_paper

    if not audio_files:
        print("❌ 'audio_files' no está definido o viene vacío. No se procesó nada.")
        return {
            "dataset": [],
            "tensors": (None, None, None),
            "torch_ds": None,
            "metadata": [],
            "summary": {"vowel": Counter(), "condition": Counter()},
        }

    dataset = process_dataset(
        audio_files=audio_files, preprocess_fn=preprocess_fn, max_files=max_files
    )

    if not dataset:
        print(
            "❌ No se pudo construir el dataset. Revisa el preprocesamiento y los archivos."
        )
        return {
            "dataset": [],
            "tensors": (None, None, None),
            "torch_ds": None,
            "metadata": [],
            "summary": {"vowel": Counter(), "condition": Counter()},
        }

    X, y_task, y_domain, metas = to_pytorch_tensors(dataset)
    torch_ds = (
        VowelSegmentsDataset(X, y_task, y_domain, metas) if X is not None else None
    )
    dist = summarize_distribution(dataset)
    print_summary(dist)

    print("\n✅ Dataset COMPLETO listo para entrenamiento con PyTorch!")
    print(f"  - Muestras totales: {len(dataset)}")
    if X is not None:
        print(f"  - Dimensiones de entrada: {tuple(X.shape)}")
        print("  - Ideal para CNN 2D")

    return {
        "dataset": dataset,
        "tensors": (X, y_task, y_domain),
        "torch_ds": torch_ds,
        "metadata": metas,
        "summary": dist,
    }


# ============================================================
# PATIENT-LEVEL UTILITIES (para CNN1D)
# ============================================================


def group_by_patient(metadata: List[SampleMeta]) -> Dict[str, List[int]]:
    """
    Agrupa índices de samples por patient_id.

    Útil para agregación patient-level en evaluación.

    Args:
        metadata: Lista de SampleMeta

    Returns:
        Dict {patient_id: [sample_indices]}
    """
    from collections import defaultdict

    patient_map = defaultdict(list)
    for idx, meta in enumerate(metadata):
        patient_map[meta.subject_id].append(idx)

    return dict(patient_map)


def speaker_independent_split(
    metadata: List[SampleMeta],
    test_size: float = 0.15,
    val_size: float = 0.176,
    random_state: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split estratificado speaker-independent.

    Asegura que ningún speaker aparezca en múltiples splits.
    Crítico para evitar data leakage en evaluación.

    Args:
        metadata: Lista de SampleMeta
        test_size: Fracción de pacientes para test
        val_size: Fracción de train_val para validation
        random_state: Seed para reproducibilidad

    Returns:
        train_idx: Índices de samples para train
        val_idx: Índices de samples para val
        test_idx: Índices de samples para test
    """
    from sklearn.model_selection import train_test_split

    # Obtener unique patient_ids con sus labels
    patient_labels = {}
    for meta in metadata:
        if meta.subject_id not in patient_labels:
            # Determinar label: 1 si 'pk' en condition, 0 si 'healthy'
            label = 1 if "pk" in meta.condition.lower() else 0
            patient_labels[meta.subject_id] = label

    patients = list(patient_labels.keys())
    labels = [patient_labels[p] for p in patients]

    # Split 1: separar test patients
    train_val_patients, test_patients = train_test_split(
        patients, test_size=test_size, stratify=labels, random_state=random_state
    )

    # Split 2: separar train/val patients
    train_val_labels = [patient_labels[p] for p in train_val_patients]
    train_patients, val_patients = train_test_split(
        train_val_patients,
        test_size=val_size,
        stratify=train_val_labels,
        random_state=random_state,
    )

    # Convertir patient lists a sample indices
    train_idx = []
    val_idx = []
    test_idx = []

    for idx, meta in enumerate(metadata):
        if meta.subject_id in train_patients:
            train_idx.append(idx)
        elif meta.subject_id in val_patients:
            val_idx.append(idx)
        elif meta.subject_id in test_patients:
            test_idx.append(idx)

    print("\n" + "=" * 70)
    print("SPEAKER-INDEPENDENT SPLIT")
    print("=" * 70)
    print(f"Pacientes únicos: {len(patients)}")
    print(f"  - Train: {len(train_patients)} pacientes → {len(train_idx)} samples")
    print(f"  - Val:   {len(val_patients)} pacientes → {len(val_idx)} samples")
    print(f"  - Test:  {len(test_patients)} pacientes → {len(test_idx)} samples")
    print("=" * 70 + "\n")

    return train_idx, val_idx, test_idx