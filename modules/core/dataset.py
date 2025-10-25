"""
Dataset Pipeline Module
========================
Pipeline completo para crear datasets PyTorch desde archivos de audio.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from collections import Counter
from pathlib import Path
import pickle
import numpy as np
import torch
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

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
    """Print detailed progress with files processed and remaining."""
    if i % max(1, every) == 0:
        processed = i + 1
        remaining = total - processed
        percentage = (processed / total) * 100
        print(f"  📁 {processed}/{total} ({percentage:.1f}%) - {path_name}")
        print(f"     ✅ Procesados: {processed} | ⏳ Faltan: {remaining}")


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
    checkpoint_path: Optional[str] = None,
    resume_from_checkpoint: bool = True,
    clear_existing_checkpoint: bool = False,
) -> List[Dict]:
    """
    Process the dataset using the paper's preprocessing function with checkpoint support.

    Args:
        audio_files: Iterable of pathlib.Path-like objects.
        preprocess_fn: Callable that returns (spectrograms, segments) per file.
        max_files: Optional cap on number of files to process.
        progress_every: Print progress every N files.
        default_sr: Sampling rate to attach to metadata (if unknown externally).
        checkpoint_path: Path to save/load checkpoint file (optional).
        resume_from_checkpoint: If True, resume from checkpoint if exists.
        clear_existing_checkpoint: If True, clear existing checkpoint and start fresh.

    Returns:
        A list of dict samples with spectrograms, segments, and metadata.
    """
    if preprocess_fn is None:
        preprocess_fn = preprocessing.preprocess_audio_paper

    if not audio_files:
        print("Error: 'audio_files' está vacío: no hay nada que procesar.")
        return []

    if max_files:
        files_to_process = list(audio_files[:max_files])
    else:
        files_to_process = list(audio_files)
    total_files = len(files_to_process)

    # Manejo de checkpoints
    dataset: List[Dict] = []
    processed_files: List[str] = []
    start_index = 0

    if checkpoint_path and not clear_existing_checkpoint:
        checkpoint_data = load_checkpoint(checkpoint_path)
        if checkpoint_data and resume_from_checkpoint:
            dataset = checkpoint_data["dataset"]
            processed_files = checkpoint_data["processed_files"]
            start_index = len(processed_files)
            print(f"🔄 Continuando desde archivo {start_index + 1}/{total_files}")
        elif checkpoint_data:
            print("⚠️ Checkpoint encontrado pero resume_from_checkpoint=False")

    if clear_existing_checkpoint and checkpoint_path:
        clear_checkpoint(checkpoint_path)
        print("🗑️ Checkpoint eliminado, comenzando desde cero")

    if start_index == 0:
        print(f"🔄 Procesando {total_files} archivos...")
    else:
        remaining = total_files - start_index
        print(f"🔄 Continuando procesamiento: {remaining} archivos restantes")
    print(f"📊 Configuración: progress_every={progress_every}")

    successful_files = len(processed_files)
    failed_files = 0

    for i, file_path in enumerate(files_to_process):
        # Saltar archivos ya procesados
        if i < start_index:
            continue

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
            successful_files += 1
        except Exception as e:
            print(f"❌ Error al procesar {file_path}: {e}. Continuando...")
            failed_files += 1
            processed_files.append(str(file_path))
            continue

        if not spectrograms:
            # Nada que agregar de este archivo
            failed_files += 1
            processed_files.append(str(file_path))
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

        # Agregar archivo a la lista de procesados
        processed_files.append(str(file_path))

        # Guardar checkpoint cada cierto número de archivos
        if checkpoint_path and (i + 1) % max(1, progress_every) == 0:
            save_checkpoint(dataset, processed_files, checkpoint_path, total_files)

    # Limpiar checkpoint al finalizar exitosamente
    if checkpoint_path:
        clear_checkpoint(checkpoint_path)
        print("✅ Procesamiento completado, checkpoint eliminado")

    # Resumen final detallado
    print("\n" + "=" * 60)
    print("📊 RESUMEN DEL PROCESAMIENTO")
    print("=" * 60)
    print(f"📁 Archivos totales: {total_files}")
    print(f"✅ Archivos exitosos: {successful_files}")
    print(f"❌ Archivos fallidos: {failed_files}")
    print(f"📈 Tasa de éxito: {(successful_files / total_files) * 100:.1f}%")
    print(f"🎯 Muestras generadas: {len(dataset)}")
    if successful_files > 0:
        print(f"📊 Promedio muestras/archivo: {len(dataset) / successful_files:.1f}")
    print("=" * 60)

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
        print("Error: Dataset vacío: no hay tensores que crear.")
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
    print("PyTorch tensors listos:")
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
    print("\nDISTRIBUCIÓN POR VOCAL:")
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
# MULTIPROCESSING UTILITIES
# ============================================================


def _process_single_file(args: Tuple) -> Tuple[Optional[Dict], str, bool]:
    """
    Procesa un solo archivo de audio. Función para multiprocessing.

    Args:
        args: Tupla con (file_path, vowel_type, preprocess_fn, default_sr)

    Returns:
        Tupla con (resultado, filename, success)
    """
    file_path, vowel_type, preprocess_fn, default_sr = args

    try:
        # Parsear metadatos del archivo
        subject_id, parsed_vowel, condition = parse_filename(
            getattr(file_path, "stem", str(file_path))
        )

        # Usar vowel_type del argumento si está disponible
        if vowel_type:
            final_vowel = vowel_type
        else:
            final_vowel = parsed_vowel

        # Procesar archivo
        spectrograms, segments = preprocess_fn(file_path, vowel_type=final_vowel)

        if not spectrograms:
            return None, str(file_path), False

        # Crear muestras
        samples = []
        for j, (spec, seg) in enumerate(zip(spectrograms, segments)):
            sample = {
                "spectrogram": spec,
                "segment": seg,
                "metadata": SampleMeta(
                    subject_id=subject_id,
                    vowel_type=final_vowel,
                    condition=condition,
                    filename=getattr(file_path, "name", str(file_path)),
                    segment_id=j,
                    sr=default_sr,
                ),
            }
            samples.append(sample)

        return samples, str(file_path), True

    except Exception as e:
        print(f"❌ Error procesando {file_path}: {e}")
        return None, str(file_path), False


def process_dataset_parallel(
    audio_files: Sequence,
    preprocess_fn: Optional[Callable] = None,
    max_files: Optional[int] = None,
    progress_every: int = 10,
    default_sr: int = 44100,
    checkpoint_path: Optional[str] = None,
    resume_from_checkpoint: bool = True,
    clear_existing_checkpoint: bool = False,
    n_workers: Optional[int] = None,
    chunk_size: int = 1,
) -> List[Dict]:
    """
    Procesa el dataset usando multiprocessing para acelerar el procesamiento.

    Args:
        audio_files: Lista de archivos de audio
        preprocess_fn: Función de preprocesamiento
        max_files: Máximo número de archivos
        progress_every: Cada cuántos archivos mostrar progreso
        default_sr: Sample rate por defecto
        checkpoint_path: Ruta del checkpoint
        resume_from_checkpoint: Si continuar desde checkpoint
        clear_existing_checkpoint: Si limpiar checkpoint existente
        n_workers: Número de procesos paralelos (None = auto)
        chunk_size: Tamaño de chunk para cada worker

    Returns:
        Dataset procesado
    """
    if preprocess_fn is None:
        preprocess_fn = preprocessing.preprocess_audio_paper

    if not audio_files:
        print("Error: 'audio_files' está vacío: no hay nada que procesar.")
        return []

    if max_files:
        files_to_process = list(audio_files[:max_files])
    else:
        files_to_process = list(audio_files)
    total_files = len(files_to_process)

    # Configurar número de workers
    if n_workers is None:
        n_workers = min(mp.cpu_count(), len(files_to_process))

    print(f"🚀 Procesamiento paralelo con {n_workers} workers")
    print(f"📊 Archivos a procesar: {total_files}")
    print(f"⚙️ Configuración: progress_every={progress_every}, chunk_size={chunk_size}")

    # Manejo de checkpoints
    dataset: List[Dict] = []
    processed_files: List[str] = []
    start_index = 0

    if checkpoint_path and not clear_existing_checkpoint:
        checkpoint_data = load_checkpoint(checkpoint_path)
        if checkpoint_data and resume_from_checkpoint:
            dataset = checkpoint_data["dataset"]
            processed_files = checkpoint_data["processed_files"]
            start_index = len(processed_files)
            print(f"🔄 Continuando desde archivo {start_index + 1}/{total_files}")
        elif checkpoint_data:
            print("⚠️ Checkpoint encontrado pero resume_from_checkpoint=False")

    if clear_existing_checkpoint and checkpoint_path:
        clear_checkpoint(checkpoint_path)
        print("🗑️ Checkpoint eliminado, comenzando desde cero")

    if start_index == 0:
        print(f"🔄 Procesando {total_files} archivos...")
    else:
        remaining = total_files - start_index
        print(f"🔄 Continuando procesamiento: {remaining} archivos restantes")

    # Preparar argumentos para multiprocessing
    files_to_process_remaining = files_to_process[start_index:]

    if not files_to_process_remaining:
        print("✅ Todos los archivos ya fueron procesados")
        return dataset

    # Crear argumentos para cada archivo
    process_args = []
    for file_path in files_to_process_remaining:
        subject_id, vowel_type, condition = parse_filename(
            getattr(file_path, "stem", str(file_path))
        )
        process_args.append((file_path, vowel_type, preprocess_fn, default_sr))

    # Procesar en paralelo
    successful_files = len(processed_files)
    failed_files = 0
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Enviar trabajos
        future_to_file = {
            executor.submit(_process_single_file, args): args[0]
            for args in process_args
        }

        # Procesar resultados conforme van completándose
        for i, future in enumerate(as_completed(future_to_file), start=start_index):
            file_path = future_to_file[future]

            try:
                result, filename, success = future.result()

                if success and result:
                    dataset.extend(result)
                    successful_files += 1
                else:
                    failed_files += 1

                processed_files.append(str(file_path))

                # Mostrar progreso
                if (i + 1) % max(1, progress_every) == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    remaining = total_files - (i + 1)
                    eta = remaining / rate if rate > 0 else 0

                    print(
                        f"  📁 {i + 1}/{total_files} ({((i + 1) / total_files) * 100:.1f}%) - {filename}"
                    )
                    print(
                        f"     ✅ Procesados: {successful_files} | ❌ Fallidos: {failed_files}"
                    )
                    print(
                        f"     ⚡ Velocidad: {rate:.1f} archivos/seg | ⏱️ ETA: {eta / 60:.1f} min"
                    )

                # Guardar checkpoint
                if checkpoint_path and (i + 1) % max(1, progress_every) == 0:
                    save_checkpoint(
                        dataset, processed_files, checkpoint_path, total_files
                    )

            except Exception as e:
                print(f"❌ Error procesando {file_path}: {e}")
                failed_files += 1
                processed_files.append(str(file_path))

    # Limpiar checkpoint al finalizar exitosamente
    if checkpoint_path:
        clear_checkpoint(checkpoint_path)
        print("✅ Procesamiento completado, checkpoint eliminado")

    # Resumen final detallado
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("📊 RESUMEN DEL PROCESAMIENTO PARALELO")
    print("=" * 60)
    print(f"📁 Archivos totales: {total_files}")
    print(f"✅ Archivos exitosos: {successful_files}")
    print(f"❌ Archivos fallidos: {failed_files}")
    print(f"📈 Tasa de éxito: {(successful_files / total_files) * 100:.1f}%")
    print(f"🎯 Muestras generadas: {len(dataset)}")
    print(f"⏱️ Tiempo total: {total_time / 60:.1f} minutos")
    print(f"⚡ Velocidad promedio: {total_files / total_time:.1f} archivos/seg")
    if successful_files > 0:
        print(f"📊 Promedio muestras/archivo: {len(dataset) / successful_files:.1f}")
    print("=" * 60)

    return dataset


# ============================================================
# CHECKPOINT UTILITIES
# ============================================================


def save_checkpoint(
    dataset: List[Dict],
    processed_files: List[str],
    checkpoint_path: str,
    total_files: int,
) -> None:
    """
    Guarda checkpoint del procesamiento para poder continuar después.

    Args:
        dataset: Dataset parcial procesado hasta ahora
        processed_files: Lista de archivos ya procesados
        checkpoint_path: Ruta donde guardar el checkpoint
        total_files: Total de archivos a procesar
    """
    checkpoint_file = Path(checkpoint_path)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_data = {
        "dataset": dataset,
        "processed_files": processed_files,
        "total_files": total_files,
        "timestamp": np.datetime64("now"),
    }

    with open(checkpoint_file, "wb") as f:
        pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = checkpoint_file.stat().st_size / (1024 * 1024)
    print(f"\n💾 Checkpoint guardado: {checkpoint_path}")
    print(f"   Tamaño: {size_mb:.1f} MB")
    print(f"   Archivos procesados: {len(processed_files)}/{total_files}")
    print(f"   Muestras en dataset: {len(dataset)}")


def load_checkpoint(checkpoint_path: str) -> Optional[Dict]:
    """
    Carga checkpoint del procesamiento.

    Args:
        checkpoint_path: Ruta del archivo checkpoint

    Returns:
        Diccionario con datos del checkpoint o None si no existe
    """
    checkpoint_file = Path(checkpoint_path)

    if not checkpoint_file.exists():
        return None

    try:
        with open(checkpoint_file, "rb") as f:
            checkpoint_data = pickle.load(f)

        size_mb = checkpoint_file.stat().st_size / (1024 * 1024)
        print(f"✅ Checkpoint cargado: {checkpoint_path}")
        print(f"   Tamaño: {size_mb:.1f} MB")
        processed_count = len(checkpoint_data["processed_files"])
        total_count = checkpoint_data["total_files"]
        print(f"   Archivos procesados: {processed_count}/{total_count}")
        print(f"   Muestras en dataset: {len(checkpoint_data['dataset'])}")

        return checkpoint_data
    except Exception as e:
        print(f"❌ Error cargando checkpoint: {e}")
        return None


def clear_checkpoint(checkpoint_path: str) -> None:
    """
    Elimina el archivo checkpoint.

    Args:
        checkpoint_path: Ruta del archivo checkpoint
    """
    checkpoint_file = Path(checkpoint_path)
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        print(f"🗑️ Checkpoint eliminado: {checkpoint_path}")


def process_dataset_with_checkpoint(
    audio_files: Sequence,
    checkpoint_path: str,
    preprocess_fn: Optional[Callable] = None,
    max_files: Optional[int] = None,
    progress_every: int = 10,
    default_sr: int = 44100,
    force_restart: bool = False,
) -> List[Dict]:
    """
    Función de conveniencia para procesar dataset con checkpoint automático.

    Args:
        audio_files: Lista de archivos de audio
        checkpoint_path: Ruta del checkpoint
        preprocess_fn: Función de preprocesamiento
        max_files: Máximo número de archivos
        progress_every: Cada cuántos archivos mostrar progreso
        default_sr: Sample rate por defecto
        force_restart: Si True, ignora checkpoint y comienza desde cero

    Returns:
        Dataset procesado
    """
    return process_dataset(
        audio_files=audio_files,
        preprocess_fn=preprocess_fn,
        max_files=max_files,
        progress_every=progress_every,
        default_sr=default_sr,
        checkpoint_path=checkpoint_path,
        resume_from_checkpoint=not force_restart,
        clear_existing_checkpoint=force_restart,
    )


def process_dataset_parallel_with_checkpoint(
    audio_files: Sequence,
    checkpoint_path: str,
    preprocess_fn: Optional[Callable] = None,
    max_files: Optional[int] = None,
    progress_every: int = 10,
    default_sr: int = 44100,
    force_restart: bool = False,
    n_workers: Optional[int] = None,
    chunk_size: int = 1,
) -> List[Dict]:
    """
    Función de conveniencia para procesar dataset en paralelo con checkpoint.

    Args:
        audio_files: Lista de archivos de audio
        checkpoint_path: Ruta del checkpoint
        preprocess_fn: Función de preprocesamiento
        max_files: Máximo número de archivos
        progress_every: Cada cuántos archivos mostrar progreso
        default_sr: Sample rate por defecto
        force_restart: Si True, ignora checkpoint y comienza desde cero
        n_workers: Número de procesos paralelos (None = auto)
        chunk_size: Tamaño de chunk para cada worker

    Returns:
        Dataset procesado
    """
    return process_dataset_parallel(
        audio_files=audio_files,
        preprocess_fn=preprocess_fn,
        max_files=max_files,
        progress_every=progress_every,
        default_sr=default_sr,
        checkpoint_path=checkpoint_path,
        resume_from_checkpoint=not force_restart,
        clear_existing_checkpoint=force_restart,
        n_workers=n_workers,
        chunk_size=chunk_size,
    )


# ============================================================
# CACHE UTILITIES
# ============================================================


def save_spectrograms_cache(dataset: List[Dict], cache_path: str) -> None:
    """
    Guarda espectrogramas individuales en cache.

    Según Ibarra et al. (2023), guardamos espectrogramas individuales (65×41)
    que pueden reutilizarse para CNN2D y Time-CNN-LSTM.

    Args:
        dataset: Lista de dicts con espectrogramas y metadata
        cache_path: Path al archivo cache
    """
    cache_file = Path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    with open(cache_file, "wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = cache_file.stat().st_size / (1024 * 1024)
    print(f"\n💾 Cache guardado: {cache_path}")
    print(f"   Tamaño: {size_mb:.1f} MB")
    print(f"   Muestras: {len(dataset)}")


def load_spectrograms_cache(cache_path: str) -> Optional[List[Dict]]:
    """
    Carga espectrogramas individuales desde cache.

    Args:
        cache_path: Path al archivo cache

    Returns:
        Lista de dicts con espectrogramas o None si no existe
    """
    cache_file = Path(cache_path)

    if not cache_file.exists():
        return None

    try:
        with open(cache_file, "rb") as f:
            dataset = pickle.load(f)

        size_mb = cache_file.stat().st_size / (1024 * 1024)
        print(f"Cache cargado: {cache_path}")
        print(f"   Tamaño: {size_mb:.1f} MB")
        print(f"   Muestras: {len(dataset)}")

        return dataset
    except Exception as e:
        print(f"Error cargando cache: {e}")
        return None


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

    # Warning si muy pocos pacientes
    if len(test_patients) < 5:
        print(f"\n⚠️  WARNING: Solo {len(test_patients)} pacientes en test!")
        print(f"   • Métricas patient-level pueden no ser representativas")
        print(f"   • Considerar usar más datos o K-fold CV")

    # Verificar que hay ambas clases en cada split
    for split_name, split_pats in [
        ("Train", train_patients),
        ("Val", val_patients),
        ("Test", test_patients),
    ]:
        labels_in_split = [patient_labels[p] for p in split_pats]
        n_hc = sum(1 for l in labels_in_split if l == 0)
        n_pd = sum(1 for l in labels_in_split if l == 1)
        if n_hc == 0 or n_pd == 0:
            print(f"   ⚠️  {split_name}: Solo tiene una clase! HC={n_hc}, PD={n_pd}")

    print("=" * 70 + "\n")

    return train_idx, val_idx, test_idx


# ============================================================
# PYTORCH DATASET WRAPPER
# ============================================================


class DictDataset(torch.utils.data.Dataset):
    """Wrapper para convertir tensores en diccionarios."""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {"spectrogram": self.X[idx], "label": self.y[idx]}
