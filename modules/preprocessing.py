"""
Audio Preprocessing Module
===========================
Implementación del preprocesamiento según paper de Domain Adaptation.

Reference:
    "Towards a Corpus (and Language)-Independent Screening of Parkinson's
    Disease from Voice and Speech through Domain Adaptation"
"""

from typing import List, Optional, Tuple
import numpy as np
import librosa


# ============================================================
# CONSTANTS (Paper Specifications)
# ============================================================

SAMPLE_RATE = 44100  # Hz
WINDOW_MS = 400  # ms
OVERLAP = 0.5  # 50%
N_MELS = 65  # Mel bands
HOP_MS = 10  # ms
FFT_WINDOW = 40  # ms para todas las vocales sostenidas
TARGET_FRAMES = 41  # frames por espectrograma

# Constantes legacy (para compatibilidad)
FFT_WINDOW_A = 40  # ms (deprecado - usar FFT_WINDOW)
FFT_WINDOW_OTHER = 40  # ms (cambiado de 25ms a 40ms)


# ============================================================
# PREPROCESSING FUNCTIONS
# ============================================================


def load_audio_file(
    file_path: str, target_sr: int = SAMPLE_RATE
) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """
    Load and resample audio file to target sample rate.

    Args:
        file_path: Path to audio file
        target_sr: Target sample rate (default: 44.1 kHz)

    Returns:
        audio: Resampled audio signal
        sr: Sample rate
    """
    try:
        audio, original_sr = librosa.load(file_path, sr=None)

        if original_sr != target_sr:
            audio = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)

        return audio, target_sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None


def segment_audio(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    window_duration: float = WINDOW_MS / 1000,
    overlap: float = OVERLAP,
) -> List[np.ndarray]:
    """
    Segment audio into overlapping windows.

    Args:
        audio: Audio signal
        sr: Sample rate
        window_duration: Window duration in seconds (default: 0.4s = 400ms)
        overlap: Overlap ratio (default: 0.5 = 50%)

    Returns:
        List of audio segments
    """
    window_samples = int(window_duration * sr)
    hop_samples = int(window_samples * (1 - overlap))

    segments = []
    for start in range(0, len(audio) - window_samples + 1, hop_samples):
        segments.append(audio[start : start + window_samples])

    return segments


def create_mel_spectrogram(
    segment: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    hop_length: Optional[int] = None,
    window_length: Optional[int] = None,
    vowel_type: str = "a",
) -> np.ndarray:
    """
    Create Mel spectrogram with paper-specific parameters.

    Args:
        segment: Audio segment
        sr: Sample rate
        n_mels: Number of Mel bands (default: 65)
        hop_length: Hop length in samples (default: 10ms)
        window_length: FFT window length (default: 40ms todas vocales)
        vowel_type: Vowel type (compatibilidad, no afecta ventana)

    Returns:
        Mel spectrogram in dB
    """
    if hop_length is None:
        hop_length = int(HOP_MS * sr / 1000)

    if window_length is None:
        # Todas las vocales sostenidas usan ventana de 40ms
        window_length = int(FFT_WINDOW * sr / 1000)

    mel_spec = librosa.feature.melspectrogram(
        y=segment,
        sr=sr,
        n_mels=n_mels,
        hop_length=hop_length,
        n_fft=window_length,
        fmax=sr // 2,
    )

    return librosa.power_to_db(mel_spec, ref=np.max)


def normalize_spectrogram(mel_spec_db: np.ndarray) -> np.ndarray:
    """
    Apply z-score normalization to spectrogram.

    Args:
        mel_spec_db: Mel spectrogram in dB

    Returns:
        Normalized spectrogram
    """
    return (mel_spec_db - np.mean(mel_spec_db)) / (np.std(mel_spec_db) + 1e-8)


def preprocess_audio_paper(
    file_path: str, vowel_type: str = "a"
) -> Tuple[Optional[List[np.ndarray]], Optional[List[np.ndarray]]]:
    """
    Complete preprocessing pipeline following the paper.

    Pipeline:
        1. Load and resample to 44.1 kHz
        2. Segment into 400ms windows with 50% overlap
        3. Create Mel spectrograms (65 bands)
        4. Normalize with z-score
        5. Ensure 65×41 dimensions

    Args:
        file_path: Path to audio file
        vowel_type: Vowel type for FFT window selection

    Returns:
        spectrograms: List of normalized Mel spectrograms (65×41)
        segments: List of audio segments
    """
    # 1. Load and resample
    audio, sr = load_audio_file(file_path)
    if audio is None:
        return None, None

    # 2. Segment
    segments = segment_audio(audio, sr=sr)

    # 3. Create Mel spectrograms
    spectrograms = []
    for segment in segments:
        mel_spec = create_mel_spectrogram(
            segment,
            sr=sr,
            n_mels=N_MELS,
            hop_length=int(HOP_MS * sr / 1000),
            vowel_type=vowel_type,
        )

        # 4-5. Normalize and adjust dimensions
        normalized_spec = normalize_spectrogram(mel_spec)
        if normalized_spec.shape[1] != TARGET_FRAMES:
            normalized_spec = librosa.util.fix_length(
                normalized_spec, size=TARGET_FRAMES, axis=1
            )

        spectrograms.append(normalized_spec)

    return spectrograms, segments


# ============================================================
# UTILITY FUNCTIONS
# ============================================================


def get_preprocessing_config() -> dict:
    """Get preprocessing configuration dictionary."""
    return {
        "SAMPLE_RATE": SAMPLE_RATE,
        "WINDOW_MS": WINDOW_MS,
        "OVERLAP": OVERLAP,
        "N_MELS": N_MELS,
        "HOP_MS": HOP_MS,
        "FFT_WINDOW": FFT_WINDOW,
        "TARGET_FRAMES": TARGET_FRAMES,
    }


def print_preprocessing_config():
    """Print preprocessing configuration."""
    config = get_preprocessing_config()
    print("⚙️ Preprocessing Configuration:")
    for key, value in config.items():
        print(f"  • {key}: {value}")
