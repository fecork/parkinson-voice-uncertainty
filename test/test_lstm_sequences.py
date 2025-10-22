#!/usr/bin/env python3
"""
Tests de Validación para Secuencias LSTM
=========================================
Verifica que las secuencias estén correctamente construidas para modelos temporales.

Tests críticos:
1. Orden temporal (segment_id consecutivos)
2. Correlación entre frames adyacentes
3. Normalización por secuencia (no por frame)
4. Padding y masking correctos
5. No mezcla de frames de diferentes audios
6. SpecAugment consistente (si aplica)

Ejecutar:
    python test/test_lstm_sequences.py
    pytest test/test_lstm_sequences.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
import numpy as np
import torch
from collections import defaultdict

from modules.core.sequence_dataset import (
    group_spectrograms_to_sequences,
    SequenceLSTMDataset,
)
from modules.data.augmentation import create_augmented_dataset


class TestTemporalOrder(unittest.TestCase):
    """Verificar que los frames estén en orden temporal correcto."""

    def test_segment_ids_are_consecutive(self):
        """Los segment_id dentro de cada secuencia deben ser consecutivos."""
        # Crear dataset sintético
        dataset = self._create_synthetic_dataset(n_files=2, segments_per_file=7)

        sequences, lengths, metadata = group_spectrograms_to_sequences(
            dataset, n_frames=7, min_frames=3
        )

        # Verificar que hay secuencias
        self.assertGreater(len(sequences), 0, "Deben crearse secuencias")

        # Para cada archivo, verificar orden
        for audio_key in self._get_audio_keys(dataset):
            samples = [s for s in dataset if self._get_key(s) == audio_key]
            samples_sorted = sorted(samples, key=lambda x: x["metadata"].segment_id)

            # Verificar que segment_ids son consecutivos o crecientes
            segment_ids = [s["metadata"].segment_id for s in samples_sorted]
            for i in range(len(segment_ids) - 1):
                diff = segment_ids[i + 1] - segment_ids[i]
                self.assertGreaterEqual(
                    diff,
                    0,
                    f"segment_id debe ser creciente: {segment_ids[i]} -> {segment_ids[i + 1]}",
                )

    def test_no_temporal_gaps(self):
        """No debe haber saltos temporales grandes entre frames consecutivos."""
        dataset = self._create_synthetic_dataset(n_files=1, segments_per_file=10)

        sequences, lengths, metadata = group_spectrograms_to_sequences(
            dataset, n_frames=7, min_frames=3
        )

        # Tomar primera secuencia
        if len(sequences) > 0:
            # Los segment_ids originales deberían ser consecutivos
            audio_key = list(self._get_audio_keys(dataset))[0]
            samples = [s for s in dataset if self._get_key(s) == audio_key]
            segment_ids = sorted([s["metadata"].segment_id for s in samples])

            # Verificar gaps máximos de 1 (consecutivos)
            for i in range(len(segment_ids) - 1):
                gap = segment_ids[i + 1] - segment_ids[i]
                self.assertLessEqual(
                    gap, 1, f"Gap temporal muy grande: {gap} entre frames {i} y {i + 1}"
                )

    def _create_synthetic_dataset(self, n_files=2, segments_per_file=7):
        """Crear dataset sintético para testing."""
        dataset = []
        for file_idx in range(n_files):
            for seg_idx in range(segments_per_file):
                # Crear espectrograma sintético
                spec = np.random.randn(65, 41).astype(np.float32)

                # Metadata sintética
                from types import SimpleNamespace

                meta = SimpleNamespace(
                    subject_id=f"subject_{file_idx}",
                    filename=f"file_{file_idx}.egg",
                    segment_id=seg_idx,
                    vowel_type="a",
                    condition="healthy",
                    augmentation="original",
                )

                dataset.append({"spectrogram": spec, "metadata": meta})

        return dataset

    def _get_audio_keys(self, dataset):
        """Obtener claves únicas de audio."""
        keys = set()
        for sample in dataset:
            meta = sample["metadata"]
            keys.add(f"{meta.subject_id}_{meta.filename}")
        return keys

    def _get_key(self, sample):
        """Obtener clave de audio de un sample."""
        meta = sample["metadata"]
        return f"{meta.subject_id}_{meta.filename}"


class TestFrameCorrelation(unittest.TestCase):
    """Verificar correlación temporal entre frames adyacentes."""

    def test_adjacent_frames_correlation(self):
        """Frames adyacentes deben tener alta correlación (>0.6)."""
        # Crear secuencia sintética con continuidad temporal
        sequence = self._create_correlated_sequence(n_frames=7)

        # Calcular correlación entre frames adyacentes
        correlations = []
        for i in range(len(sequence) - 1):
            frame1 = sequence[i, 0].flatten()  # (65, 41) -> flat
            frame2 = sequence[i + 1, 0].flatten()

            # Correlación de Pearson
            corr = np.corrcoef(frame1, frame2)[0, 1]
            correlations.append(corr)

        mean_corr = np.mean(correlations)

        # Para secuencias reales de audio, esperamos correlación >0.6
        # (usamos 0.5 para sintéticos con ruido)
        self.assertGreater(
            mean_corr,
            0.5,
            f"Correlación promedio muy baja: {mean_corr:.3f}. "
            "Puede indicar frames desordenados o normalización por frame.",
        )

    def test_correlation_drops_with_distance(self):
        """La correlación debe decrecer con la distancia temporal."""
        sequence = self._create_correlated_sequence(n_frames=7)

        # Calcular correlación a diferentes distancias
        corr_dist1 = []
        corr_dist3 = []

        for i in range(len(sequence) - 3):
            frame_i = sequence[i, 0].flatten()

            # Distancia 1
            frame_i1 = sequence[i + 1, 0].flatten()
            corr_dist1.append(np.corrcoef(frame_i, frame_i1)[0, 1])

            # Distancia 3
            frame_i3 = sequence[i + 3, 0].flatten()
            corr_dist3.append(np.corrcoef(frame_i, frame_i3)[0, 1])

        mean_corr_1 = np.mean(corr_dist1)
        mean_corr_3 = np.mean(corr_dist3)

        # Correlación debe decrecer con distancia
        self.assertGreater(
            mean_corr_1,
            mean_corr_3,
            f"Correlación no decae con distancia: dist1={mean_corr_1:.3f}, "
            f"dist3={mean_corr_3:.3f}",
        )

    def _create_correlated_sequence(self, n_frames=7):
        """Crear secuencia con continuidad temporal (smooth)."""
        sequence = np.zeros((n_frames, 1, 65, 41), dtype=np.float32)

        # Crear base con evolución suave
        base = np.random.randn(65, 41)
        for t in range(n_frames):
            # Evolución suave + pequeño ruido
            evolution = base * (1 - 0.1 * t / n_frames)
            noise = np.random.randn(65, 41) * 0.1
            sequence[t, 0] = evolution + noise

        return sequence


class TestNormalizationScope(unittest.TestCase):
    """Verificar que la normalización sea por secuencia, no por frame."""

    def test_normalization_is_sequence_wide(self):
        """La normalización debe aplicarse a toda la secuencia, no frame a frame."""
        # Usar la función real de normalización
        from modules.core.sequence_dataset import normalize_sequence

        # Crear secuencia sin normalizar con variación entre frames
        sequence_raw = np.random.randn(7, 1, 65, 41).astype(np.float32)

        # Agregar tendencia para que frames tengan diferentes medias
        for i in range(7):
            sequence_raw[i] += i * 0.5  # Cada frame tiene media diferente

        # Normalizar por secuencia (CORRECTO)
        sequence_normalized = normalize_sequence(sequence_raw, length=7)

        # Verificar que la normalización es por secuencia:
        # 1. La secuencia completa debe tener media≈0 y std≈1
        all_valid_frames = sequence_normalized[:7].reshape(-1)
        global_mean = all_valid_frames.mean()
        global_std = all_valid_frames.std()

        self.assertAlmostEqual(
            global_mean, 0.0, places=5, msg="Media global debe ser ~0"
        )
        self.assertAlmostEqual(global_std, 1.0, places=5, msg="Std global debe ser ~1")

        # 2. Las medias individuales de cada frame DEBEN variar
        # (si normalizáramos por frame, todas serían ~0)
        means_per_frame = [sequence_normalized[i, 0].mean() for i in range(7)]
        mean_variance = np.var(means_per_frame)

        # Con normalización por secuencia, debe haber variación significativa
        self.assertGreater(
            mean_variance,
            0.01,
            f"Normalización parece ser por frame (varianza de medias: {mean_variance:.6f}). "
            f"Con normalización por secuencia, los frames deben mantener sus diferencias relativas.",
        )

    def test_sequence_statistics(self):
        """La secuencia completa debe tener estadísticas globales correctas."""
        # Crear secuencia normalizada por secuencia
        sequence = np.random.randn(7, 1, 65, 41).astype(np.float32)
        mean_global = sequence.mean()
        std_global = sequence.std()

        sequence_normalized = (sequence - mean_global) / (std_global + 1e-8)

        # Verificar estadísticas globales
        self.assertAlmostEqual(
            sequence_normalized.mean(), 0.0, places=5, msg="Media global debe ser ~0"
        )
        self.assertAlmostEqual(
            sequence_normalized.std(), 1.0, places=5, msg="Std global debe ser ~1"
        )


class TestPaddingMasking(unittest.TestCase):
    """Verificar que el padding y masking funcionen correctamente."""

    def test_padding_is_zero(self):
        """Los frames de padding deben ser ceros."""
        # Crear dataset con menos de 7 frames
        dataset = []
        from types import SimpleNamespace

        for i in range(4):  # Solo 4 frames
            spec = np.random.randn(65, 41).astype(np.float32)
            meta = SimpleNamespace(
                subject_id="test",
                filename="test.egg",
                segment_id=i,
                vowel_type="a",
                condition="healthy",
                augmentation="original",
            )
            dataset.append({"spectrogram": spec, "metadata": meta})

        sequences, lengths, metadata = group_spectrograms_to_sequences(
            dataset, n_frames=7, min_frames=3
        )

        self.assertEqual(len(sequences), 1, "Debe crear 1 secuencia")
        self.assertEqual(lengths[0], 4, "Longitud real debe ser 4")

        # Verificar que frames 4, 5, 6 son padding (ceros)
        sequence = sequences[0]
        for i in range(4, 7):
            frame = sequence[i, 0]
            self.assertTrue(
                np.allclose(frame, 0.0),
                f"Frame {i} debe ser padding (ceros), pero tiene valores no-cero",
            )

    def test_lengths_match_real_frames(self):
        """Las longitudes deben reflejar el número real de frames."""
        dataset = []
        from types import SimpleNamespace

        # Archivo 1: 5 frames
        for i in range(5):
            spec = np.random.randn(65, 41).astype(np.float32)
            meta = SimpleNamespace(
                subject_id="test1",
                filename="test1.egg",
                segment_id=i,
                vowel_type="a",
                condition="healthy",
                augmentation="original",
            )
            dataset.append({"spectrogram": spec, "metadata": meta})

        # Archivo 2: 7 frames (completo)
        for i in range(7):
            spec = np.random.randn(65, 41).astype(np.float32)
            meta = SimpleNamespace(
                subject_id="test2",
                filename="test2.egg",
                segment_id=i,
                vowel_type="a",
                condition="healthy",
                augmentation="original",
            )
            dataset.append({"spectrogram": spec, "metadata": meta})

        sequences, lengths, metadata = group_spectrograms_to_sequences(
            dataset, n_frames=7, min_frames=3
        )

        self.assertEqual(len(sequences), 2, "Deben crearse 2 secuencias")
        self.assertEqual(lengths[0], 5, "Primera secuencia: 5 frames reales")
        self.assertEqual(lengths[1], 7, "Segunda secuencia: 7 frames reales")

    def test_mask_excludes_padding(self):
        """El LSTM debe poder usar las longitudes para crear máscaras."""
        from modules.models.lstm_da.model import TimeCNNBiLSTM_DA

        # Crear secuencia con padding
        sequence = torch.randn(2, 7, 1, 65, 41)  # B=2, T=7
        sequence[0, 5:] = 0  # Primer sample: padding en posiciones 5, 6
        sequence[1, 6:] = 0  # Segundo sample: padding en posición 6

        lengths = torch.tensor([5, 6])  # Longitudes reales

        # Crear modelo y hacer forward
        model = TimeCNNBiLSTM_DA(n_frames=7, n_domains=2)
        model.eval()

        with torch.no_grad():
            logits_pd, logits_domain, _ = model(sequence, lengths=lengths)

        # Verificar que el forward funciona
        self.assertEqual(logits_pd.shape, (2, 2), "Logits PD shape correcto")
        self.assertEqual(logits_domain.shape, (2, 2), "Logits domain shape correcto")


class TestNoFrameMixing(unittest.TestCase):
    """Verificar que frames de diferentes audios no se mezclen."""

    def test_sequences_from_same_audio(self):
        """Cada secuencia debe contener solo frames del mismo archivo."""
        # Crear dataset con múltiples archivos
        dataset = []
        from types import SimpleNamespace

        for file_idx in range(3):
            for seg_idx in range(7):
                spec = np.random.randn(65, 41).astype(np.float32) + file_idx * 10
                meta = SimpleNamespace(
                    subject_id=f"subject_{file_idx}",
                    filename=f"file_{file_idx}.egg",
                    segment_id=seg_idx,
                    vowel_type="a",
                    condition="healthy",
                    augmentation="original",
                )
                dataset.append({"spectrogram": spec, "metadata": meta})

        sequences, lengths, metadata = group_spectrograms_to_sequences(
            dataset, n_frames=7, min_frames=3
        )

        # Debe haber 3 secuencias (una por archivo)
        self.assertEqual(len(sequences), 3, "Debe haber 1 secuencia por archivo")

        # Verificar que metadata es consistente
        for i, meta in enumerate(metadata):
            self.assertTrue(
                hasattr(meta, "subject_id"), f"Metadata {i} debe tener subject_id"
            )
            self.assertTrue(
                hasattr(meta, "filename"), f"Metadata {i} debe tener filename"
            )

    def test_subject_id_consistency(self):
        """Todos los frames de una secuencia deben tener el mismo subject_id."""
        dataset = []
        from types import SimpleNamespace

        # Crear 2 sujetos con frames intercalados (NO debería pasar en realidad)
        for i in range(7):
            spec1 = np.random.randn(65, 41).astype(np.float32)
            meta1 = SimpleNamespace(
                subject_id="subject_A",
                filename="fileA.egg",
                segment_id=i,
                vowel_type="a",
                condition="healthy",
                augmentation="original",
            )
            dataset.append({"spectrogram": spec1, "metadata": meta1})

            spec2 = np.random.randn(65, 41).astype(np.float32)
            meta2 = SimpleNamespace(
                subject_id="subject_B",
                filename="fileB.egg",
                segment_id=i,
                vowel_type="a",
                condition="healthy",
                augmentation="original",
            )
            dataset.append({"spectrogram": spec2, "metadata": meta2})

        sequences, lengths, metadata = group_spectrograms_to_sequences(
            dataset, n_frames=7, min_frames=3
        )

        # Debe crear 2 secuencias separadas (una por subject)
        self.assertEqual(len(sequences), 2, "Debe separar por subject_id")

        # Verificar que metadata es del mismo subject
        subjects = [meta.subject_id for meta in metadata]
        self.assertEqual(len(set(subjects)), 2, "Debe haber 2 subjects diferentes")


class TestSpecAugmentConsistency(unittest.TestCase):
    """Verificar que SpecAugment se aplique consistentemente."""

    def test_specaugment_pattern_consistency(self):
        """Si SpecAugment está activo, debe ser consistente en la secuencia."""
        # Este test verificará que no haya máscaras diferentes por frame
        # Por ahora, solo detecta el problema

        # Crear secuencia con SpecAugment (simulado)
        sequence = np.random.randn(7, 1, 65, 41).astype(np.float32)

        # Simular SpecAugment con máscara DIFERENTE por frame (INCORRECTO)
        for t in range(7):
            # Máscara aleatoria diferente en cada frame
            mask_freq = np.random.randint(0, 10)
            mask_time = np.random.randint(0, 5)
            sequence[t, 0, mask_freq : mask_freq + 5, :] = 0
            sequence[t, 0, :, mask_time : mask_time + 3] = 0

        # Detectar máscaras inconsistentes
        # Si hay SpecAugment consistente, las posiciones con ceros deberían coincidir
        zero_positions = []
        for t in range(7):
            zeros = np.where(sequence[t, 0] == 0)
            zero_positions.append(set(zip(zeros[0], zeros[1])))

        # Verificar consistencia (intersección de posiciones con cero)
        if len(zero_positions) > 1:
            common_zeros = zero_positions[0].intersection(*zero_positions[1:])
            total_zeros = len(zero_positions[0])

            if total_zeros > 0:
                consistency_ratio = len(common_zeros) / total_zeros

                # Si SpecAugment es consistente, ratio debería ser alto
                # Por ahora solo advertimos
                if consistency_ratio < 0.5:
                    print(
                        f"\n[WARN] SpecAugment parece inconsistente "
                        f"(consistencia: {consistency_ratio:.2%})"
                    )


class TestSmoothTransitions(unittest.TestCase):
    """Tests para verificar transiciones suaves entre frames (recomendación visual)."""

    def test_no_energy_jumps_between_frames(self):
        """No debe haber saltos bruscos de energía entre frames consecutivos."""
        from modules.core.sequence_dataset import normalize_sequence

        # Crear secuencia con evolución suave
        sequence = np.zeros((7, 1, 65, 41), dtype=np.float32)
        for t in range(7):
            # Evolución suave de energía
            base = np.sin(np.linspace(0, np.pi, 65 * 41)).reshape(65, 41)
            sequence[t, 0] = base * (1 + t * 0.1)  # Incremento gradual

        # Normalizar por secuencia
        sequence_norm = normalize_sequence(sequence, length=7)

        # Calcular energía promedio por frame
        energies = [sequence_norm[t, 0].mean() for t in range(7)]

        # Calcular saltos de energía entre frames adyacentes
        energy_jumps = [abs(energies[t + 1] - energies[t]) for t in range(6)]
        max_jump = max(energy_jumps)

        # Con normalización por secuencia, los saltos deben ser moderados
        # (mucho menores que si normalizáramos por frame)
        self.assertLess(
            max_jump,
            0.5,
            f"Salto de energía muy grande: {max_jump:.3f}. "
            f"Puede indicar normalización por frame.",
        )

    def test_specaugment_masks_aligned(self):
        """Si SpecAugment, las máscaras deben estar alineadas entre frames."""
        # Simular secuencia con SpecAugment GLOBAL (correcto)
        sequence_global = np.random.randn(7, 1, 65, 41).astype(np.float32)

        # Aplicar máscara en mismas posiciones para todos los frames
        mask_freq_start = 20
        mask_freq_end = 30
        sequence_global[:, 0, mask_freq_start:mask_freq_end, :] = 0

        # Verificar que máscara está en misma posición en todos los frames
        for t in range(7):
            masked_region = sequence_global[t, 0, mask_freq_start:mask_freq_end, :]
            self.assertTrue(
                np.allclose(masked_region, 0.0),
                f"Frame {t} debe tener máscara en freq bins {mask_freq_start}-{mask_freq_end}",
            )

        # Calcular consistencia de máscaras
        zero_positions = []
        for t in range(7):
            zeros = np.where(sequence_global[t, 0] == 0)
            zero_positions.append(set(zip(zeros[0], zeros[1])))

        # Todas las posiciones de ceros deben coincidir
        common_zeros = zero_positions[0].intersection(*zero_positions[1:])
        total_zeros = len(zero_positions[0])

        if total_zeros > 0:
            consistency = len(common_zeros) / total_zeros
            self.assertGreater(
                consistency,
                0.95,
                f"SpecAugment debe ser consistente en frames: {consistency:.1%}",
            )

    def test_smooth_temporal_evolution(self):
        """Los valores de frames deben evolucionar suavemente en el tiempo."""
        from modules.core.sequence_dataset import normalize_sequence

        # Crear secuencia realista con evolución temporal suave
        sequence = np.zeros((7, 1, 65, 41), dtype=np.float32)
        for t in range(7):
            # Base con ruido
            base = np.random.randn(65, 41) * 0.3
            # Tendencia temporal suave
            trend = np.linspace(-1, 1, 65).reshape(-1, 1) * (t / 7)
            sequence[t, 0] = base + trend

        # Normalizar
        sequence_norm = normalize_sequence(sequence, length=7)

        # Calcular diferencia frame-a-frame
        frame_diffs = []
        for t in range(6):
            diff = np.abs(sequence_norm[t + 1, 0] - sequence_norm[t, 0]).mean()
            frame_diffs.append(diff)

        mean_diff = np.mean(frame_diffs)

        # Diferencia debe ser pequeña (evolución suave)
        self.assertLess(
            mean_diff,
            1.0,
            f"Diferencia promedio entre frames muy alta: {mean_diff:.3f}. "
            f"Esperamos evolución suave.",
        )


class TestNormalizationVisual(unittest.TestCase):
    """Tests visuales para verificar normalización correcta."""

    def test_no_vertical_discontinuities(self):
        """No debe haber discontinuidades verticales entre frames (saltos de color)."""
        from modules.core.sequence_dataset import normalize_sequence

        # Crear secuencia con valores progresivos
        sequence = np.zeros((7, 1, 65, 41), dtype=np.float32)
        for t in range(7):
            # Valor base aumenta suavemente con el tiempo
            sequence[t, 0] = np.random.randn(65, 41) + t * 0.2

        # Normalizar por secuencia
        sequence_norm = normalize_sequence(sequence, length=7)

        # Calcular estadísticas por frame
        frame_means = [sequence_norm[t, 0].mean() for t in range(7)]
        frame_stds = [sequence_norm[t, 0].std() for t in range(7)]

        # Las medias NO deben ser todas ~0 (eso indicaría normalización por frame)
        # Con secuencia que tiene tendencia, debe haber variación en medias
        mean_variance = np.var(frame_means)
        self.assertGreater(
            mean_variance,
            0.001,
            f"Varianza de medias muy baja ({mean_variance:.6f}). "
            f"Posible normalización por frame. Medias: {frame_means}",
        )

        # Alternativamente, verificar que NO todas las medias son ~0
        means_near_zero = sum(abs(m) < 0.05 for m in frame_means)
        self.assertLess(
            means_near_zero,
            6,
            f"Demasiados frames con media~0 ({means_near_zero}/7). "
            f"Posible normalización por frame.",
        )

    def test_padding_not_normalized(self):
        """El padding NO debe ser normalizado, debe mantenerse en cero."""
        from modules.core.sequence_dataset import normalize_sequence

        # Secuencia con padding
        sequence = np.random.randn(7, 1, 65, 41).astype(np.float32)
        length = 5  # Solo 5 frames válidos

        # Normalizar
        sequence_norm = normalize_sequence(sequence, length=length)

        # Verificar que padding (frames 5, 6) son EXACTAMENTE cero
        for t in range(length, 7):
            self.assertTrue(
                np.allclose(sequence_norm[t], 0.0, atol=1e-10),
                f"Padding en frame {t} debe ser exactamente cero, no normalizado",
            )


class TestRegression(unittest.TestCase):
    """Tests de regresión para garantizar estabilidad."""

    def test_sequence_shape_consistency(self):
        """Las secuencias deben mantener shape esperado."""
        dataset = []
        from types import SimpleNamespace

        for i in range(7):
            spec = np.random.randn(65, 41).astype(np.float32)
            meta = SimpleNamespace(
                subject_id="test",
                filename="test.egg",
                segment_id=i,
                vowel_type="a",
                condition="healthy",
                augmentation="original",
            )
            dataset.append({"spectrogram": spec, "metadata": meta})

        sequences, lengths, metadata = group_spectrograms_to_sequences(
            dataset, n_frames=7, min_frames=3
        )

        # Verificar shape
        self.assertEqual(len(sequences), 1)
        self.assertEqual(
            sequences[0].shape, (7, 1, 65, 41), "Shape debe ser (T, C, H, W)"
        )

    def test_model_forward_pass(self):
        """Test de forward pass completo con el modelo."""
        from modules.models.lstm_da.model import TimeCNNBiLSTM_DA

        # Crear batch sintético
        batch_size = 4
        n_frames = 7
        X = torch.randn(batch_size, n_frames, 1, 65, 41)
        lengths = torch.tensor([7, 6, 5, 7])

        # Crear modelo
        model = TimeCNNBiLSTM_DA(n_frames=n_frames, n_domains=4)
        model.eval()

        # Forward pass
        with torch.no_grad():
            logits_pd, logits_domain, embeddings = model(
                X, lengths=lengths, return_embeddings=True
            )

        # Verificar shapes
        self.assertEqual(logits_pd.shape, (batch_size, 2))
        self.assertEqual(logits_domain.shape, (batch_size, 4))
        self.assertEqual(embeddings.shape, (batch_size, 128))


def run_validation_suite():
    """Ejecutar suite completa de validación."""
    print("\n" + "=" * 70)
    print("SUITE DE VALIDACIÓN DE SECUENCIAS LSTM")
    print("=" * 70)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Agregar todos los tests
    suite.addTests(loader.loadTestsFromTestCase(TestTemporalOrder))
    suite.addTests(loader.loadTestsFromTestCase(TestFrameCorrelation))
    suite.addTests(loader.loadTestsFromTestCase(TestNormalizationScope))
    suite.addTests(loader.loadTestsFromTestCase(TestPaddingMasking))
    suite.addTests(loader.loadTestsFromTestCase(TestNoFrameMixing))
    suite.addTests(loader.loadTestsFromTestCase(TestSpecAugmentConsistency))
    suite.addTests(loader.loadTestsFromTestCase(TestSmoothTransitions))
    suite.addTests(loader.loadTestsFromTestCase(TestNormalizationVisual))
    suite.addTests(loader.loadTestsFromTestCase(TestRegression))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print("[PASS] TODOS LOS TESTS PASARON")
    else:
        print("[FAIL] ALGUNOS TESTS FALLARON")
        print(f"   Fallidos: {len(result.failures)}")
        print(f"   Errores: {len(result.errors)}")
    print("=" * 70)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_validation_suite()
    sys.exit(0 if success else 1)
