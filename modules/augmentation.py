"""
Data Augmentation Module
=========================
Técnicas de augmentation para audio y espectrogramas.

Augmentation strategies:
    - Audio domain: time stretching, pitch shifting, noise injection
    - Spectrogram domain: SpecAugment, masking, mixup
"""

from typing import Tuple, Optional, Union
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
        n_steps: Number of semitones to shift (positive = up, negative = down)

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
    return augmented / np.max(np.abs(augmented))  # Normalize


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
        noise_audio = np.tile(noise_audio, int(np.ceil(len(audio) / len(noise_audio))))
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
