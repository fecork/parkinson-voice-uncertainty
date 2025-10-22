"""
Data Augmentation Module
=========================
Técnicas de augmentation para audio y espectrogramas.

Augmentation strategies:
    - Audio domain: time stretching, pitch shifting, noise injection
    - Spectrogram domain: SpecAugment, masking, mixup
"""

from typing import Tuple, Optional
import numpy as np
import librosa


# ============================================================
# AUDIO-DOMAIN AUGMENTATION
# ============================================================


def time_stretch(audio: np.ndarray, rate: float = 1.1) -> np.ndarray:
    """
    Apply time stretching without changing pitch.

    Args:
        audio: Audio signal
        rate: Stretch factor (>1 faster, <1 slower)

    Returns:
        Time-stretched audio
    """
    return librosa.effects.time_stretch(audio, rate=rate)


def pitch_shift(audio: np.ndarray, sr: int, n_steps: float = 2.0) -> np.ndarray:
    """
    Shift pitch without changing tempo.

    Args:
        audio: Audio signal
        sr: Sample rate
        n_steps: Number of semitones to shift
                 (positive = up, negative = down)

    Returns:
        Pitch-shifted audio
    """
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)


def add_white_noise(audio: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
    """
    Add white noise to audio signal.

    Args:
        audio: Audio signal
        noise_factor: Noise amplitude factor (default: 0.005)

    Returns:
        Audio with added noise
    """
    noise = np.random.randn(len(audio))
    augmented = audio + noise_factor * noise
    # Normalize
    return augmented / np.max(np.abs(augmented))


def add_background_noise(
    audio: np.ndarray, noise_audio: np.ndarray, snr_db: float = 20
) -> np.ndarray:
    """
    Add background noise with specific SNR.

    Args:
        audio: Clean audio signal
        noise_audio: Noise signal
        snr_db: Signal-to-noise ratio in dB

    Returns:
        Audio with background noise
    """
    # Ensure noise is same length as audio
    if len(noise_audio) < len(audio):
        repeats = int(np.ceil(len(audio) / len(noise_audio)))
        noise_audio = np.tile(noise_audio, repeats)
    noise_audio = noise_audio[: len(audio)]

    # Calculate signal and noise power
    signal_power = np.mean(audio**2)
    noise_power = np.mean(noise_audio**2)

    # Calculate required noise power for target SNR
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    noise_factor = np.sqrt(target_noise_power / noise_power)

    return audio + noise_factor * noise_audio


def dynamic_range_compression(audio: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Apply dynamic range compression.

    Args:
        audio: Audio signal
        threshold: Compression threshold

    Returns:
        Compressed audio
    """
    compressed = np.copy(audio)
    mask = np.abs(compressed) > threshold
    compressed[mask] = threshold + (compressed[mask] - threshold) * 0.5
    return compressed


# ============================================================
# SPECTROGRAM-DOMAIN AUGMENTATION
# ============================================================


def spec_augment(
    spectrogram: np.ndarray,
    freq_mask_param: int = 10,
    time_mask_param: int = 5,
    num_freq_masks: int = 2,
    num_time_masks: int = 1,
    p: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Apply SpecAugment with temporal coherence for LSTM.

    Applies frequency and time masking to FULL spectrogram,
    ensuring consistent masks across subsequent frames.

    Reference:
        Park et al. "SpecAugment: A Simple Data Augmentation Method for ASR"

    Args:
        spectrogram: Input spectrogram [n_mels, T]
        freq_mask_param: Maximum frequency mask size
        time_mask_param: Maximum time mask size
        num_freq_masks: Number of frequency masks
        num_time_masks: Number of time masks
        p: Probability of applying augmentation
        rng: Random number generator for reproducibility

    Returns:
        Augmented spectrogram with consistent masking
    """
    if rng is None:
        rng = np.random.default_rng()

    if rng.random() > p:
        return spectrogram

    spec_aug = spectrogram.copy()
    n_mels, T = spec_aug.shape

    # Frequency masking (horizontal bands)
    for _ in range(num_freq_masks):
        f = rng.integers(1, min(freq_mask_param + 1, n_mels))
        f0 = rng.integers(0, max(1, n_mels - f + 1))
        spec_aug[f0 : f0 + f, :] = 0.0

    # Time masking (vertical bands)
    for _ in range(num_time_masks):
        t = rng.integers(1, min(time_mask_param + 1, T))
        t0 = rng.integers(0, max(1, T - t + 1))
        spec_aug[:, t0 : t0 + t] = 0.0

    return spec_aug


def mixup_spectrograms(
    spec1: np.ndarray, spec2: np.ndarray, alpha: float = 0.4
) -> Tuple[np.ndarray, float]:
    """
    Apply mixup augmentation to spectrograms.

    Reference:
        Zhang et al. "mixup: Beyond Empirical Risk Minimization"

    Args:
        spec1: First spectrogram
        spec2: Second spectrogram
        alpha: Beta distribution parameter

    Returns:
        Mixed spectrogram and mixing coefficient lambda
    """
    lam = np.random.beta(alpha, alpha)
    mixed_spec = lam * spec1 + (1 - lam) * spec2
    return mixed_spec, lam


def random_erasing(
    spectrogram: np.ndarray,
    probability: float = 0.5,
    area_ratio_range: Tuple[float, float] = (0.02, 0.4),
    aspect_ratio_range: Tuple[float, float] = (0.3, 3.3),
) -> np.ndarray:
    """
    Apply random erasing augmentation.

    Args:
        spectrogram: Input spectrogram
        probability: Probability of applying augmentation
        area_ratio_range: Range of erased area ratio
        aspect_ratio_range: Range of aspect ratio

    Returns:
        Augmented spectrogram
    """
    if np.random.random() > probability:
        return spectrogram

    spec_aug = np.copy(spectrogram)
    freq, time = spec_aug.shape
    area = freq * time

    target_area = np.random.uniform(*area_ratio_range) * area
    aspect_ratio = np.random.uniform(*aspect_ratio_range)

    h = int(np.sqrt(target_area * aspect_ratio))
    w = int(np.sqrt(target_area / aspect_ratio))

    if h < freq and w < time:
        top = np.random.randint(0, freq - h)
        left = np.random.randint(0, time - w)
        spec_aug[top : top + h, left : left + w] = 0

    return spec_aug


# ============================================================
# COMBINED AUGMENTATION PIPELINE
# ============================================================


def augment_audio(
    audio: np.ndarray, sr: int, aug_params: Optional[dict] = None
) -> np.ndarray:
    """
    Apply random audio augmentation pipeline.

    Args:
        audio: Audio signal
        sr: Sample rate
        aug_params: Augmentation parameters

    Returns:
        Augmented audio
    """
    if aug_params is None:
        aug_params = {
            "time_stretch": True,
            "pitch_shift": True,
            "add_noise": True,
        }

    augmented = np.copy(audio)

    if aug_params.get("time_stretch", False) and np.random.random() > 0.5:
        rate = np.random.uniform(0.9, 1.1)
        augmented = time_stretch(augmented, rate=rate)

    if aug_params.get("pitch_shift", False) and np.random.random() > 0.5:
        n_steps = np.random.uniform(-2, 2)
        augmented = pitch_shift(augmented, sr=sr, n_steps=n_steps)

    if aug_params.get("add_noise", False) and np.random.random() > 0.5:
        noise_factor = np.random.uniform(0.001, 0.01)
        augmented = add_white_noise(augmented, noise_factor=noise_factor)

    return augmented


def augment_spectrogram(
    spectrogram: np.ndarray, aug_params: Optional[dict] = None
) -> np.ndarray:
    """
    Apply random spectrogram augmentation pipeline.

    Args:
        spectrogram: Input spectrogram
        aug_params: Augmentation parameters

    Returns:
        Augmented spectrogram
    """
    if aug_params is None:
        aug_params = {
            "spec_augment": True,
            "random_erasing": True,
        }

    augmented = np.copy(spectrogram)

    if aug_params.get("spec_augment", False) and np.random.random() > 0.5:
        augmented = spec_augment(
            augmented,
            freq_mask_param=10,
            time_mask_param=5,
            num_freq_masks=2,
            num_time_masks=2,
        )

    if aug_params.get("random_erasing", False) and np.random.random() > 0.5:
        augmented = random_erasing(
            augmented, probability=0.5, area_ratio_range=(0.02, 0.2)
        )

    return augmented


# ============================================================
# UTILITY FUNCTIONS
# ============================================================


def get_default_aug_config() -> dict:
    """Get default augmentation configuration."""
    return {
        "audio": {
            "time_stretch": True,
            "time_stretch_range": (0.9, 1.1),
            "pitch_shift": True,
            "pitch_shift_range": (-2, 2),
            "add_noise": True,
            "noise_factor_range": (0.001, 0.01),
        },
        "spectrogram": {
            "spec_augment": True,
            "freq_mask_param": 10,
            "time_mask_param": 5,
            "random_erasing": True,
            "erasing_probability": 0.5,
        },
    }


def print_augmentation_config():
    """Print augmentation configuration."""
    config = get_default_aug_config()
    print("🎨 Augmentation Configuration:")
    print("\n  Audio Domain:")
    for key, value in config["audio"].items():
        print(f"    • {key}: {value}")
    print("\n  Spectrogram Domain:")
    for key, value in config["spectrogram"].items():
        print(f"    • {key}: {value}")


# ============================================================
# INTEGRATED AUGMENTATION PIPELINE
# ============================================================


def preprocess_audio_with_augmentation(
    file_path: str,
    vowel_type: str = "a",
    apply_spec_augment: bool = False,
    spec_augment_params: Optional[dict] = None,
    spec_augment_before_segment: bool = True,
    skip_normalization: bool = False,
    augmentation_type: str = "spec_augment",
):
    """
    Pipeline de preprocesamiento con augmentation configurable.

    PIPELINE ORDER:
    1. Load audio completo
    2. Aplicar augmentation de audio (si corresponde)
    3. Crear espectrograma (completo o por segmentos)
    4. Aplicar SpecAugment (si corresponde)
    5. Segmentar si es necesario
    6. Normalizar (opcional)

    Args:
        file_path: Path to audio file
        vowel_type: Vowel type for FFT window selection
        apply_spec_augment: Si True, aplica SpecAugment
        spec_augment_params: Parámetros para SpecAugment
                            (freq_mask_param, time_mask_param)
        spec_augment_before_segment: Si True, aplica SpecAugment ANTES
                                     de segmentar (DEFAULT para LSTM)
        skip_normalization: Si True, no normaliza
                           (útil para normalizar por secuencia)
        augmentation_type: Tipo de augmentation a aplicar
                          ("pitch_shift", "time_stretch", "noise", "spec_augment")

    Returns:
        spectrograms: List of Mel spectrograms
        segments: List of audio segments (o None)
        aug_label: Label describing augmentation applied
    """
    from ..core import preprocessing

    # 1. Load audio completo
    audio, sr = preprocessing.load_audio_file(file_path)
    if audio is None:
        return None, None, None

    # 2. Aplicar augmentation de audio según tipo
    if augmentation_type == "pitch_shift":
        n_steps = np.random.uniform(-2, 2)
        audio = pitch_shift(audio, sr=sr, n_steps=n_steps)
        aug_label = "pitch_shift"
    elif augmentation_type == "time_stretch":
        rate = np.random.uniform(0.9, 1.1)
        audio = time_stretch(audio, rate=rate)
        aug_label = "time_stretch"
    elif augmentation_type == "noise":
        noise_factor = np.random.uniform(0.001, 0.01)
        audio = add_white_noise(audio, noise_factor=noise_factor)
        aug_label = "noise"
    elif augmentation_type == "spec_augment":
        # Para SpecAugment, no aplicamos augmentation de audio
        aug_label = "original"
    else:
        aug_label = "original"

    # 3. Pipeline: SpecAugment ANTES o DESPUÉS de segmentar
    if spec_augment_before_segment and apply_spec_augment:
        # NUEVO: Pipeline para LSTM (SpecAugment antes de segmentar)
        # Crear espectrograma completo
        full_mel_spec = preprocessing.create_full_mel_spectrogram(
            audio,
            sr=sr,
            n_mels=preprocessing.N_MELS,
            hop_length=int(preprocessing.HOP_MS * sr / 1000),
        )

        # Aplicar SpecAugment sobre espectrograma completo
        if spec_augment_params is None:
            spec_augment_params = {
                "freq_mask_param": 8,  # Conservador: ~10-12% de 65 bins
                "time_mask_param": 4,  # Conservador
                "num_freq_masks": 2,
                "num_time_masks": 1,
            }
        rng = np.random.default_rng()
        full_mel_spec = spec_augment(
            full_mel_spec,
            freq_mask_param=spec_augment_params.get("freq_mask_param", 8),
            time_mask_param=spec_augment_params.get("time_mask_param", 4),
            num_freq_masks=spec_augment_params.get("num_freq_masks", 2),
            num_time_masks=spec_augment_params.get("num_time_masks", 1),
            rng=rng,
        )

        if aug_label == "original":
            aug_label = "spec_aug_global"
        else:
            aug_label += "_spec_aug_global"

        # Segmentar espectrograma augmentado
        spectrograms = preprocessing.segment_spectrogram(
            full_mel_spec,
            target_frames=preprocessing.TARGET_FRAMES,
            overlap=preprocessing.OVERLAP,
        )

        # Normalizar si se requiere
        if not skip_normalization:
            spectrograms = [
                preprocessing.normalize_spectrogram(spec) for spec in spectrograms
            ]

        segments = None  # No hay segmentos de audio individuales

    else:
        # LEGACY: Pipeline original (segmenta primero, luego augment)
        segments = preprocessing.segment_audio(audio, sr=sr)

        spectrograms = []
        for segment in segments:
            mel_spec = preprocessing.create_mel_spectrogram(
                segment,
                sr=sr,
                n_mels=preprocessing.N_MELS,
                hop_length=int(preprocessing.HOP_MS * sr / 1000),
                vowel_type=vowel_type,
            )

            # SpecAugment por frame (INCONSISTENTE para LSTM)
            if apply_spec_augment:
                if spec_augment_params is None:
                    spec_augment_params = {
                        "freq_mask_param": 10,
                        "time_mask_param": 5,
                        "num_freq_masks": 2,
                        "num_time_masks": 2,
                    }
                mel_spec = spec_augment(mel_spec, **spec_augment_params)
                if aug_label == "original":
                    aug_label = "spec_aug"
                else:
                    aug_label += "_spec_aug"

            # Normalizar si se requiere
            if not skip_normalization:
                mel_spec = preprocessing.normalize_spectrogram(mel_spec)

            if mel_spec.shape[1] != preprocessing.TARGET_FRAMES:
                mel_spec = librosa.util.fix_length(
                    mel_spec, size=preprocessing.TARGET_FRAMES, axis=1
                )

            spectrograms.append(mel_spec)

    return spectrograms, segments, aug_label


def _add_samples_to_dataset(
    all_samples: list,
    spectrograms: list,
    segments: list,
    aug_label: str,
    subject_id: str,
    vowel_type: str,
    condition: str,
    filename: str,
    dataset_module,
):
    """Helper to avoid code duplication when adding samples."""
    if spectrograms:
        # Handle case where segments is None (SpecAugment before segmentation)
        if segments is None:
            segments = [None] * len(spectrograms)

        for j, (spec, seg) in enumerate(zip(spectrograms, segments)):
            all_samples.append(
                {
                    "spectrogram": spec,
                    "segment": seg,
                    "metadata": dataset_module.SampleMeta(
                        subject_id=subject_id,
                        vowel_type=vowel_type,
                        condition=condition,
                        filename=filename,
                        segment_id=j,
                        sr=44100,
                    ),
                    "augmentation": aug_label,
                }
            )


def create_augmented_dataset(
    audio_files,
    augmentation_types: list = None,
    apply_spec_augment: bool = True,
    num_spec_augment_versions: int = 2,
    spec_augment_before_segment: bool = True,
    skip_normalization: bool = False,
    progress_every: int = 5,
    use_cache: bool = True,
    cache_dir: str = "./cache",
    force_regenerate: bool = False,
):
    """
    Crea dataset con augmentation configurable.

    PIPELINE:
    1. Load audio
    2. Crear espectrograma Mel
    3. Aplicar augmentation según configuración
    4. Segmentar (si spec_augment_before_segment=True)
    5. Normalizar (opcional)

    Args:
        audio_files: Lista de archivos de audio
        augmentation_types: Lista de tipos de augmentation a aplicar
                           ["original", "pitch_shift", "time_stretch", "noise", "spec_augment"]
                           Por defecto: ["original", "spec_augment"]
        apply_spec_augment: DEPRECATED - usar augmentation_types en su lugar
        num_spec_augment_versions: Versiones de SpecAugment por muestra
            (e.g., 2 = original + 2 versiones augmentadas)
        spec_augment_before_segment: Si True, aplica ANTES de segmentar
                                     (DEFAULT para LSTM - coherencia temporal)
        skip_normalization: Si True, no normaliza
                           (para normalizar por secuencia)
        progress_every: Frecuencia de progreso
        use_cache: Si True, usa cache
        cache_dir: Directorio de cache
        force_regenerate: Si True, regenera cache

    Returns:
        dataset: Lista de muestras con augmentation aplicada
    """
    from ..core import dataset as dataset_module, preprocessing
    import os
    import pickle
    import hashlib

    # ============================================================
    # CONFIGURACIÓN DE AUGMENTATION
    # ============================================================

    # Configuración por defecto: solo SpecAugment
    if augmentation_types is None:
        augmentation_types = ["original", "spec_augment"]

    # Compatibilidad con parámetro deprecated
    if not apply_spec_augment and "spec_augment" in augmentation_types:
        augmentation_types = [t for t in augmentation_types if t != "spec_augment"]
    elif apply_spec_augment and "spec_augment" not in augmentation_types:
        augmentation_types.append("spec_augment")

    # Validar tipos de augmentation
    valid_types = ["original", "pitch_shift", "time_stretch", "noise", "spec_augment"]
    invalid_types = [t for t in augmentation_types if t not in valid_types]
    if invalid_types:
        raise ValueError(
            f"Tipos de augmentation inválidos: {invalid_types}. Válidos: {valid_types}"
        )

    # Determinar qué augmentation aplicar
    apply_pitch_shift = "pitch_shift" in augmentation_types
    apply_time_stretch = "time_stretch" in augmentation_types
    apply_noise = "noise" in augmentation_types
    apply_spec_augment_final = "spec_augment" in augmentation_types

    # ============================================================
    # CACHE MANAGEMENT
    # ============================================================
    if use_cache:
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)

        # Generate unique cache key
        aug_types_str = "_".join(sorted(augmentation_types))
        cache_key_data = (
            f"{len(audio_files)}_aug_{aug_types_str}_"
            f"v{num_spec_augment_versions}_before{spec_augment_before_segment}_"
            f"skipnorm{skip_normalization}"
        )
        cache_key = hashlib.md5(cache_key_data.encode()).hexdigest()[:8]
        cache_file = os.path.join(cache_dir, f"augmented_dataset_{cache_key}.pkl")

        # Try to load from cache
        if os.path.exists(cache_file) and not force_regenerate:
            print("💾 Cargando dataset desde cache...")
            print("   📁 " + cache_file)
            try:
                with open(cache_file, "rb") as f:
                    all_samples = pickle.load(f)
                n_samples = len(all_samples)
                print(f"✅ Cache cargado exitosamente: {n_samples} muestras")
                time_saved = len(audio_files) * 0.5
                print(f"⚡ Tiempo ahorrado: ~{time_saved:.1f} min")
                return all_samples
            except Exception as e:
                print(f"⚠️  Error leyendo cache: {e}")
                print("   Regenerando dataset...")

    # ============================================================
    # GENERATE DATASET (if cache miss or disabled)
    # ============================================================
    print("[INFO] Creando dataset con augmentation configurable...")
    n_files = len(audio_files)
    print(f"   Archivos: {n_files}")
    print(f"   [CONFIG] Tipos de augmentation: {augmentation_types}")

    if apply_pitch_shift:
        print("   [CONFIG] Pitch Shift: ACTIVADO")
    if apply_time_stretch:
        print("   [CONFIG] Time Stretch: ACTIVADO")
    if apply_noise:
        print("   [CONFIG] Noise Injection: ACTIVADO")
    if apply_spec_augment_final:
        print(
            f"   [CONFIG] SpecAugment: ACTIVADO ({num_spec_augment_versions} versiones)"
        )
        if spec_augment_before_segment:
            print(
                "   [CONFIG] SpecAugment GLOBAL (antes de segmentar) - LSTM optimizado"
            )
        else:
            print("   [CONFIG] SpecAugment por frame - CNN legacy")

    all_samples = []

    for i, file_path in enumerate(audio_files):
        if i % progress_every == 0:
            fname = getattr(file_path, "name", str(file_path))
            print(f"  [{i + 1}/{len(audio_files)}] {fname}")

        # Parse metadata
        subject_id, vowel_type_parsed, condition = dataset_module.parse_filename(
            getattr(file_path, "stem", str(file_path))
        )
        filename = getattr(file_path, "name", str(file_path))

        # Aplicar augmentation según configuración
        try:
            # 1. Original (si está en la lista)
            if "original" in augmentation_types:
                specs, segs = preprocessing.preprocess_audio_paper(
                    file_path, vowel_type=vowel_type_parsed
                )
                _add_samples_to_dataset(
                    all_samples,
                    specs,
                    segs,
                    "original",
                    subject_id,
                    vowel_type_parsed,
                    condition,
                    filename,
                    dataset_module,
                )

            # 2. Pitch Shift
            if apply_pitch_shift:
                try:
                    result = preprocess_audio_with_augmentation(
                        file_path,
                        vowel_type=vowel_type_parsed,
                        apply_spec_augment=False,
                        spec_augment_params=None,
                        spec_augment_before_segment=spec_augment_before_segment,
                        skip_normalization=skip_normalization,
                        augmentation_type="pitch_shift",
                    )
                    if result[0] is not None:
                        specs_aug, segs_aug, _ = result
                        _add_samples_to_dataset(
                            all_samples,
                            specs_aug,
                            segs_aug,
                            "pitch_shift",
                            subject_id,
                            vowel_type_parsed,
                            condition,
                            filename,
                            dataset_module,
                        )
                except Exception as e:
                    print(f"    [WARN] Error Pitch Shift: {e}")

            # 3. Time Stretch
            if apply_time_stretch:
                try:
                    result = preprocess_audio_with_augmentation(
                        file_path,
                        vowel_type=vowel_type_parsed,
                        apply_spec_augment=False,
                        spec_augment_params=None,
                        spec_augment_before_segment=spec_augment_before_segment,
                        skip_normalization=skip_normalization,
                        augmentation_type="time_stretch",
                    )
                    if result[0] is not None:
                        specs_aug, segs_aug, _ = result
                        _add_samples_to_dataset(
                            all_samples,
                            specs_aug,
                            segs_aug,
                            "time_stretch",
                            subject_id,
                            vowel_type_parsed,
                            condition,
                            filename,
                            dataset_module,
                        )
                except Exception as e:
                    print(f"    [WARN] Error Time Stretch: {e}")

            # 4. Noise Injection
            if apply_noise:
                try:
                    result = preprocess_audio_with_augmentation(
                        file_path,
                        vowel_type=vowel_type_parsed,
                        apply_spec_augment=False,
                        spec_augment_params=None,
                        spec_augment_before_segment=spec_augment_before_segment,
                        skip_normalization=skip_normalization,
                        augmentation_type="noise",
                    )
                    if result[0] is not None:
                        specs_aug, segs_aug, _ = result
                        _add_samples_to_dataset(
                            all_samples,
                            specs_aug,
                            segs_aug,
                            "noise",
                            subject_id,
                            vowel_type_parsed,
                            condition,
                            filename,
                            dataset_module,
                        )
                except Exception as e:
                    print(f"    [WARN] Error Noise Injection: {e}")

            # 5. SpecAugment (múltiples versiones)
            if apply_spec_augment_final and num_spec_augment_versions > 0:
                for spec_ver in range(num_spec_augment_versions):
                    try:
                        result = preprocess_audio_with_augmentation(
                            file_path,
                            vowel_type=vowel_type_parsed,
                            apply_spec_augment=True,
                            spec_augment_params=None,
                            spec_augment_before_segment=spec_augment_before_segment,
                            skip_normalization=skip_normalization,
                            augmentation_type="spec_augment",
                        )
                        if result[0] is None:
                            continue

                        specs_aug, segs_aug, _ = result
                        aug_label = f"spec_aug_v{spec_ver + 1}"
                        _add_samples_to_dataset(
                            all_samples,
                            specs_aug,
                            segs_aug,
                            aug_label,
                            subject_id,
                            vowel_type_parsed,
                            condition,
                            filename,
                            dataset_module,
                        )
                    except Exception as e:
                        print(f"    [WARN] Error SpecAugment v{spec_ver + 1}: {e}")

        except Exception as e:
            print(f"    [ERROR] Error procesando {filename}: {e}")

    print(f"\n[OK] Dataset: {len(all_samples)} muestras totales")

    # ============================================================
    # SAVE TO CACHE
    # ============================================================
    if use_cache and all_samples:
        print("💾 Guardando dataset en cache...")
        print("   📁 " + cache_file)
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(all_samples, f, protocol=pickle.HIGHEST_PROTOCOL)
            file_size_mb = os.path.getsize(cache_file) / (1024 * 1024)
            print(f"✅ Cache guardado: {file_size_mb:.1f} MB")
            print("💡 Próxima ejecución será instantánea!")
        except Exception as e:
            print(f"⚠️  Error guardando cache: {e}")

    return all_samples
