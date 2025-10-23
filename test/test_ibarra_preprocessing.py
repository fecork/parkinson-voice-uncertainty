#!/usr/bin/env python3
"""
Pruebas Unitarias - Preprocesamiento Ibarra et al. (2023)
===========================================================
Valida que el preprocesamiento cumple exactamente con el paper:

1. Resample a 44.1 kHz
2. Normalización por amplitud máxima absoluta
3. Segmentación: 400ms ventanas, 50% overlap
4. Mel spectrogram: 65 bandas, ventana FFT 40ms, hop 10ms
5. Conversión a dB
6. Normalización z-score por espectrograma individual
7. Dimensión final: 65×41 píxeles
8. Sin augmentation
"""

import sys
from pathlib import Path
import unittest
import numpy as np
import librosa

# Agregar módulos al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.core import preprocessing
from modules.core.dataset import process_dataset, to_pytorch_tensors


class TestIbarraPreprocessing(unittest.TestCase):
    """Pruebas unitarias del preprocesamiento según Ibarra et al. (2023)."""

    def setUp(self):
        """Configurar datos de prueba."""
        # Buscar un archivo de audio para pruebas
        data_path = Path("./data/vowels_healthy")
        audio_files = list(data_path.glob("*.egg"))

        if not audio_files:
            self.skipTest("No hay archivos de audio disponibles para pruebas")

        self.test_file = audio_files[0]

    def test_sample_rate_constant(self):
        """Verificar que la constante SAMPLE_RATE es 44100 Hz."""
        self.assertEqual(
            preprocessing.SAMPLE_RATE,
            44100,
            "SAMPLE_RATE debe ser 44100 Hz según paper",
        )

    def test_window_duration_constant(self):
        """Verificar que la ventana de segmentación es 400ms."""
        self.assertEqual(
            preprocessing.WINDOW_MS, 400, "WINDOW_MS debe ser 400ms según paper"
        )

    def test_overlap_constant(self):
        """Verificar que el overlap es 50%."""
        self.assertEqual(
            preprocessing.OVERLAP, 0.5, "OVERLAP debe ser 0.5 (50%) según paper"
        )

    def test_n_mels_constant(self):
        """Verificar que hay 65 bandas Mel."""
        self.assertEqual(
            preprocessing.N_MELS, 65, "N_MELS debe ser 65 bandas según paper"
        )

    def test_hop_ms_constant(self):
        """Verificar que el hop es 10ms."""
        self.assertEqual(preprocessing.HOP_MS, 10, "HOP_MS debe ser 10ms según paper")

    def test_fft_window_constant(self):
        """Verificar que la ventana FFT es 40ms para vocales sostenidas."""
        self.assertEqual(
            preprocessing.FFT_WINDOW,
            40,
            "FFT_WINDOW debe ser 40ms para vocales sostenidas según paper",
        )

    def test_target_frames_constant(self):
        """Verificar que el número de frames objetivo es 41."""
        self.assertEqual(
            preprocessing.TARGET_FRAMES, 41, "TARGET_FRAMES debe ser 41 según paper"
        )

    def test_audio_normalization_max_abs(self):
        """Verificar normalización por amplitud máxima absoluta."""
        audio, sr = preprocessing.load_audio_file(self.test_file)

        self.assertIsNotNone(audio, "Audio debe cargarse correctamente")

        # Verificar que está normalizado (max abs <= 1.0)
        max_abs = np.max(np.abs(audio))
        self.assertLessEqual(
            max_abs, 1.0, "Audio debe estar normalizado por max-abs (valor <= 1.0)"
        )

        # Verificar que realmente hay normalización (no todo ceros)
        self.assertGreater(max_abs, 0.0, "Audio no debe ser silencio")

    def test_audio_resample_44100(self):
        """Verificar que el audio se resamplea a 44.1 kHz."""
        audio, sr = preprocessing.load_audio_file(self.test_file)

        self.assertEqual(sr, 44100, "Sample rate debe ser 44100 Hz")

    def test_segmentation_400ms_50overlap(self):
        """Verificar segmentación de 400ms con 50% overlap."""
        # Crear audio sintético de 1 segundo
        sr = 44100
        duration = 1.0  # segundo
        audio = np.random.randn(int(sr * duration))

        segments = preprocessing.segment_audio(
            audio,
            sr=sr,
            window_duration=0.4,  # 400ms
            overlap=0.5,  # 50%
        )

        # Verificar que hay segmentos
        self.assertGreater(len(segments), 0, "Debe haber al menos un segmento")

        # Verificar duración de cada segmento
        expected_samples = int(sr * 0.4)  # 400ms
        for seg in segments:
            self.assertEqual(
                len(seg),
                expected_samples,
                f"Cada segmento debe tener {expected_samples} samples (400ms a 44.1kHz)",
            )

        # Verificar overlap del 50%
        # Con 1s de audio y ventanas de 400ms con 50% overlap:
        # Esperamos: (1000 - 400) / (400 * 0.5) + 1 = 4 segmentos
        expected_segments = int((duration * 1000 - 400) / (400 * 0.5) + 1)
        self.assertEqual(
            len(segments),
            expected_segments,
            f"Con 1s audio, 400ms ventanas y 50% overlap → {expected_segments} segmentos",
        )

    def test_mel_spectrogram_dimensions(self):
        """Verificar que los espectrogramas tienen dimensión 65×41."""
        spectrograms, _ = preprocessing.preprocess_audio_paper(self.test_file)

        self.assertIsNotNone(spectrograms, "Debe generar espectrogramas")
        self.assertGreater(len(spectrograms), 0, "Debe haber al menos un espectrograma")

        for spec in spectrograms:
            self.assertEqual(
                spec.shape,
                (65, 41),
                "Espectrograma debe tener dimensión 65×41 (bandas Mel × frames)",
            )

    def test_z_score_normalization(self):
        """Verificar normalización z-score."""
        spectrograms, _ = preprocessing.preprocess_audio_paper(self.test_file)

        for spec in spectrograms:
            # Verificar que la media está cerca de 0
            mean = np.mean(spec)
            self.assertAlmostEqual(
                mean,
                0.0,
                places=5,
                msg="Media debe estar cerca de 0 después de z-score",
            )

            # Verificar que la desviación estándar está cerca de 1
            std = np.std(spec)
            self.assertAlmostEqual(
                std,
                1.0,
                places=5,
                msg="Desviación estándar debe estar cerca de 1 después de z-score",
            )

    def test_no_augmentation(self):
        """Verificar que NO se aplica augmentation."""
        # Procesar el mismo archivo dos veces
        specs1, _ = preprocessing.preprocess_audio_paper(self.test_file)
        specs2, _ = preprocessing.preprocess_audio_paper(self.test_file)

        self.assertEqual(
            len(specs1),
            len(specs2),
            "Mismo archivo debe producir mismo número de espectrogramas",
        )

        # Verificar que los espectrogramas son idénticos (sin variación aleatoria)
        for s1, s2 in zip(specs1, specs2):
            np.testing.assert_array_almost_equal(
                s1,
                s2,
                decimal=10,
                err_msg="Espectrogramas deben ser idénticos (sin augmentation)",
            )

    def test_hop_length_10ms(self):
        """Verificar que hop length corresponde a 10ms."""
        sr = 44100
        hop_ms = 10
        expected_hop_samples = int(hop_ms * sr / 1000)

        # El hop length esperado es 441 samples (10ms a 44.1kHz)
        self.assertEqual(
            expected_hop_samples,
            441,
            "Hop length debe ser 441 samples (10ms a 44.1kHz)",
        )

        # Verificar que la constante en el módulo es correcta
        actual_hop = int(preprocessing.HOP_MS * preprocessing.SAMPLE_RATE / 1000)
        self.assertEqual(
            actual_hop,
            expected_hop_samples,
            f"HOP_MS debe resultar en {expected_hop_samples} samples",
        )

    def test_fft_window_40ms(self):
        """Verificar que ventana FFT corresponde a 40ms."""
        sr = 44100
        fft_ms = 40
        expected_fft_samples = int(fft_ms * sr / 1000)

        # La ventana FFT esperada es 1764 samples (40ms a 44.1kHz)
        self.assertEqual(
            expected_fft_samples,
            1764,
            "Ventana FFT debe ser 1764 samples (40ms a 44.1kHz)",
        )

        # Verificar constante en el módulo
        actual_fft = int(preprocessing.FFT_WINDOW * preprocessing.SAMPLE_RATE / 1000)
        self.assertEqual(
            actual_fft,
            expected_fft_samples,
            f"FFT_WINDOW debe resultar en {expected_fft_samples} samples",
        )

    def test_pipeline_integration(self):
        """Test de integración del pipeline completo."""
        # Procesar un archivo completo
        dataset = process_dataset(
            audio_files=[self.test_file],
            preprocess_fn=preprocessing.preprocess_audio_paper,
            max_files=1,
        )

        self.assertGreater(
            len(dataset), 0, "Dataset debe contener al menos una muestra"
        )

        # Verificar estructura de cada muestra
        for sample in dataset:
            self.assertIn("spectrogram", sample, "Debe tener espectrograma")
            self.assertIn("segment", sample, "Debe tener segmento de audio")
            self.assertIn("metadata", sample, "Debe tener metadata")

            # Verificar dimensiones
            spec = sample["spectrogram"]
            self.assertEqual(
                spec.shape, (65, 41), "Espectrograma en dataset debe ser 65×41"
            )


class TestIbarraMathematicalProperties(unittest.TestCase):
    """Pruebas matemáticas del preprocesamiento."""

    def test_mel_scale_properties(self):
        """Verificar propiedades de la escala Mel."""
        # Mel scale debe ser aproximadamente lineal hasta 1 kHz
        # y logarítmica después

        # Fórmula: mel = 2595 * log10(1 + f/700)
        freq_1khz = 1000
        mel_1khz = 2595 * np.log10(1 + freq_1khz / 700)

        # Verificar que 1 kHz corresponde a aproximadamente 1000 mel
        # Nota: valor exacto es ~1000 mel según implementación de librosa
        self.assertAlmostEqual(
            mel_1khz, 1000.0, delta=150, msg="1 kHz debe estar cerca de 1000 mel"
        )

    def test_db_conversion_properties(self):
        """Verificar propiedades de conversión a dB."""
        # dB = 10 * log10(power)
        # Verificar que valores positivos en power dan dB positivos

        power = 100
        db = 10 * np.log10(power)
        self.assertEqual(db, 20.0, "100 en power → 20 dB")

        power = 0.01
        db = 10 * np.log10(power)
        self.assertEqual(db, -20.0, "0.01 en power → -20 dB")

    def test_z_score_formula(self):
        """Verificar fórmula de z-score."""
        # z = (x - mean) / std

        data = np.array([1, 2, 3, 4, 5], dtype=float)
        mean = np.mean(data)
        std = np.std(data)

        z_score = (data - mean) / std

        # Verificar que z-score tiene media 0 y std 1
        self.assertAlmostEqual(np.mean(z_score), 0.0, places=10)
        self.assertAlmostEqual(np.std(z_score), 1.0, places=10)


def suite():
    """Crear suite de pruebas."""
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestIbarraPreprocessing))
    suite.addTest(unittest.makeSuite(TestIbarraMathematicalProperties))
    return suite


if __name__ == "__main__":
    # Ejecutar pruebas
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite())

    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN DE VALIDACIÓN - IBARRA ET AL. (2023)")
    print("=" * 70)
    print(f"Tests ejecutados: {result.testsRun}")
    print(f"Exitosos: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Fallidos: {len(result.failures)}")
    print(f"Errores: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✅ TODOS LOS TESTS PASARON")
        print("   El preprocesamiento cumple con el paper de Ibarra et al. (2023)")
    else:
        print("\n❌ ALGUNOS TESTS FALLARON")
        print("   Revisar implementación del preprocesamiento")

    print("=" * 70)

    sys.exit(0 if result.wasSuccessful() else 1)
