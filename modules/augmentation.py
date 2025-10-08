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
    num_time_masks: int = 2,
) -> np.ndarray:
    """
    Apply SpecAugment (frequency and time masking).

    Reference:
        Park et al. "SpecAugment: A Simple Data Augmentation Method for ASR"

    Args:
        spectrogram: Input spectrogram (freq x time)
        freq_mask_param: Maximum frequency mask size
        time_mask_param: Maximum time mask size
        num_freq_masks: Number of frequency masks
        num_time_masks: Number of time masks

    Returns:
        Augmented spectrogram
    """
    spec_aug = np.copy(spectrogram)
    freq, time = spec_aug.shape

    # Frequency masking
    for _ in range(num_freq_masks):
        f = np.random.randint(0, freq_mask_param)
        f0 = np.random.randint(0, freq - f)
        spec_aug[f0 : f0 + f, :] = 0

    # Time masking
    for _ in range(num_time_masks):
        t = np.random.randint(0, time_mask_param)
        t0 = np.random.randint(0, time - t)
        spec_aug[:, t0 : t0 + t] = 0

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
    aug_type: str = "original",
    aug_params: Optional[dict] = None,
    apply_spec_augment: bool = False,
    spec_augment_params: Optional[dict] = None,
):
    """
    Complete preprocessing pipeline WITH augmentation.

    PIPELINE ORDER (CORRECTO):
    1. Load audio completo
    2. Apply augmentation → AUDIO COMPLETO (pitch/time/noise)
    3. Segment into 400ms windows
    4. Create Mel spectrograms (25ms/40ms FFT)
    5. Apply SpecAugment (opcional) → SOBRE ESPECTROGRAMA
    6. Normalize

    Args:
        file_path: Path to audio file
        vowel_type: Vowel type for FFT window selection
        aug_type: Type of augmentation to apply:
                  - "original": no augmentation
                  - "pitch_shift": pitch shifting
                  - "time_stretch": time stretching
                  - "noise": add white noise
                  - "combined": all three
        aug_params: Specific parameters for augmentation
                   (pitch_steps, stretch_rate, noise_factor)
        apply_spec_augment: Whether to apply SpecAugment
        spec_augment_params: Parameters for SpecAugment
                            (freq_mask_param, time_mask_param)

    Returns:
        spectrograms: List of normalized Mel spectrograms
        segments: List of audio segments
        aug_label: Label describing augmentation applied
    """
    from . import preprocessing

    # 1. Load audio completo
    audio, sr = preprocessing.load_audio_file(file_path)
    if audio is None:
        return None, None, None

    aug_label = "original"

    # 2. Apply augmentation AL AUDIO COMPLETO (antes de segmentar)
    if aug_type != "original" and aug_params is not None:
        if aug_type == "pitch_shift":
            n_steps = aug_params.get("n_steps", 2)
            audio = pitch_shift(audio, sr=sr, n_steps=n_steps)
            aug_label = f"pitch_{n_steps:+d}"

        elif aug_type == "time_stretch":
            rate = aug_params.get("rate", 1.1)
            audio = time_stretch(audio, rate=rate)
            aug_label = f"time_{rate:.2f}x"

        elif aug_type == "noise":
            noise_factor = aug_params.get("noise_factor", 0.005)
            audio = add_white_noise(audio, noise_factor=noise_factor)
            aug_label = f"noise_{noise_factor:.4f}"

        elif aug_type == "combined":
            # Apply all three
            n_steps = aug_params.get("n_steps", 1)
            rate = aug_params.get("rate", 1.05)
            noise_factor = aug_params.get("noise_factor", 0.003)

            audio = pitch_shift(audio, sr=sr, n_steps=n_steps)
            audio = time_stretch(audio, rate=rate)
            audio = add_white_noise(audio, noise_factor=noise_factor)
            aug_label = f"combined_{n_steps:+d}_{rate:.2f}_{noise_factor:.4f}"

    # 3. Segment (DESPUÉS de augmentation de audio)
    segments = preprocessing.segment_audio(audio, sr=sr)

    # 4-5-6. Create Mel spectrograms, apply SpecAugment, normalize
    spectrograms = []
    for segment in segments:
        mel_spec = preprocessing.create_mel_spectrogram(
            segment,
            sr=sr,
            n_mels=preprocessing.N_MELS,
            hop_length=int(preprocessing.HOP_MS * sr / 1000),
            vowel_type=vowel_type,
        )

        # 5. Apply SpecAugment (opcional, DESPUÉS de crear espectrograma)
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

        # 6. Normalize
        normalized_spec = preprocessing.normalize_spectrogram(mel_spec)
        if normalized_spec.shape[1] != preprocessing.TARGET_FRAMES:
            normalized_spec = librosa.util.fix_length(
                normalized_spec, size=preprocessing.TARGET_FRAMES, axis=1
            )

        spectrograms.append(normalized_spec)

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
    apply_spec_augment: bool = False,
    progress_every: int = 5,
):
    """
    Create a dataset with multiple augmentation versions.

    Aplica augmentation en 2 niveles:
    NIVEL 1 - Audio (antes de segmentar):
      1. Pitch shifting: ±1, ±2 semitonos
      2. Time stretching: 0.9x, 1.1x
      3. Ruido aditivo: 0.005 (≈30 dB SNR)

    NIVEL 2 - Espectrograma (después de Mel):
      4. SpecAugment: frequency + time masking

    Args:
        audio_files: List of audio file paths
        augmentation_types: List with:
            ["original", "pitch_shift", "time_stretch", "noise"]
            Default: all four types
        apply_spec_augment: If True, apply SpecAugment to all samples
        progress_every: Print progress frequency

    Returns:
        dataset: List of augmented samples
    """
    from . import dataset as dataset_module, preprocessing

    if augmentation_types is None:
        augmentation_types = ["original", "pitch_shift", "time_stretch", "noise"]

    # Configuración de augmentation
    AUG_CONFIGS = {
        "pitch_shift": [("pitch_shift", {"n_steps": n}) for n in [-2, -1, 1, 2]],
        "time_stretch": [("time_stretch", {"rate": r}) for r in [0.9, 1.1]],
        "noise": [("noise", {"noise_factor": 0.005})],
    }

    print("🎨 Creando dataset con augmentation...")
    n_files = len(audio_files)
    types_preview = augmentation_types[:2]
    print(f"   Archivos: {n_files} | Tipos: {types_preview}...")
    if apply_spec_augment:
        print("   ✨ SpecAugment: ACTIVADO")

    all_samples = []

    for i, file_path in enumerate(audio_files):
        if i % progress_every == 0:
            fname = getattr(file_path, "name", str(file_path))
            print(f"  {i + 1}/{len(audio_files)}: {fname}")

        # Parse metadata una sola vez
        subject_id, vowel_type, condition = dataset_module.parse_filename(
            getattr(file_path, "stem", str(file_path))
        )
        filename = getattr(file_path, "name", str(file_path))

        # Original
        if "original" in augmentation_types:
            try:
                specs, segs = preprocessing.preprocess_audio_paper(
                    file_path, vowel_type=vowel_type
                )
                _add_samples_to_dataset(
                    all_samples,
                    specs,
                    segs,
                    "original",
                    subject_id,
                    vowel_type,
                    condition,
                    filename,
                    dataset_module,
                )
            except Exception as e:
                print(f"    ⚠️  Error original: {e}")

        # Procesar augmentations configuradas
        for aug_type in augmentation_types:
            if aug_type in AUG_CONFIGS:
                for aug_name, aug_params in AUG_CONFIGS[aug_type]:
                    try:
                        specs, segs, aug_label = preprocess_audio_with_augmentation(
                            file_path,
                            vowel_type=vowel_type,
                            aug_type=aug_name,
                            aug_params=aug_params,
                            apply_spec_augment=apply_spec_augment,
                        )
                        _add_samples_to_dataset(
                            all_samples,
                            specs,
                            segs,
                            aug_label,
                            subject_id,
                            vowel_type,
                            condition,
                            filename,
                            dataset_module,
                        )
                    except Exception as e:
                        params_str = f"{aug_params}"
                        print(f"    ⚠️  Error {aug_name} {params_str}: {e}")

    print(f"\n✅ Dataset: {len(all_samples)} muestras totales")
    return all_samples
