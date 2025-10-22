"""
Pruebas Unitarias para Time-CNN-BiLSTM-DA
==========================================
Verifica que la implementación matemática y arquitectónica sea correcta.
"""

import sys
from pathlib import Path
import unittest

import torch
import torch.nn as nn
import numpy as np

# Agregar path para imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.models.lstm_da.model import TimeCNNBiLSTM_DA
from modules.core.sequence_dataset import group_spectrograms_to_sequences


class TestSequenceGeneration(unittest.TestCase):
    """Tests para generación de secuencias con padding."""

    def setUp(self):
        """Setup: crear dataset de prueba."""
        # Crear dataset mock con diferentes números de espectrogramas por archivo
        self.dataset = []
        
        # Archivo 1: 10 espectrogramas (suficientes para n_frames=7)
        for i in range(10):
            spec = np.random.randn(65, 41).astype(np.float32)
            meta = type('obj', (object,), {
                'subject_id': 'subj_1',
                'filename': 'audio_1.wav',
                'segment_id': i,
                'vowel_type': 'a',
                'condition': 'h',
            })()
            self.dataset.append({
                'spectrogram': spec,
                'metadata': meta
            })
        
        # Archivo 2: 5 espectrogramas (necesita padding para n_frames=7)
        for i in range(5):
            spec = np.random.randn(65, 41).astype(np.float32)
            meta = type('obj', (object,), {
                'subject_id': 'subj_2',
                'filename': 'audio_2.wav',
                'segment_id': i,
                'vowel_type': 'i',
                'condition': 'l',
            })()
            self.dataset.append({
                'spectrogram': spec,
                'metadata': meta
            })
        
        # Archivo 3: 2 espectrogramas (muy pocos, se descarta si min_frames=3)
        for i in range(2):
            spec = np.random.randn(65, 41).astype(np.float32)
            meta = type('obj', (object,), {
                'subject_id': 'subj_3',
                'filename': 'audio_3.wav',
                'segment_id': i,
                'vowel_type': 'u',
                'condition': 'n',
            })()
            self.dataset.append({
                'spectrogram': spec,
                'metadata': meta
            })

    def test_sequence_shapes(self):
        """Test: Verificar que las secuencias tienen la forma correcta."""
        n_frames = 7
        sequences, lengths, metas = group_spectrograms_to_sequences(
            self.dataset,
            n_frames=n_frames,
            min_frames=3
        )
        
        # Verificar que se crearon 2 secuencias (archivo 3 se descarta)
        self.assertEqual(len(sequences), 2)
        self.assertEqual(len(lengths), 2)
        self.assertEqual(len(metas), 2)
        
        # Verificar shape de cada secuencia
        for seq in sequences:
            self.assertEqual(seq.shape, (n_frames, 1, 65, 41))
    
    def test_padding_zeros(self):
        """Test: Verificar que el padding contiene ceros."""
        n_frames = 7
        sequences, lengths, metas = group_spectrograms_to_sequences(
            self.dataset,
            n_frames=n_frames,
            min_frames=3
        )
        
        # Secuencia 2 tiene solo 5 frames reales, debería tener padding
        seq2 = sequences[1]  # audio_2.wav con 5 frames
        length2 = lengths[1]
        
        self.assertEqual(length2, 5)
        
        # Frames 0-4 deberían tener datos (no todos ceros)
        for i in range(length2):
            self.assertFalse(np.allclose(seq2[i], 0.0))
        
        # Frames 5-6 deberían ser padding (todos ceros)
        for i in range(length2, n_frames):
            self.assertTrue(np.allclose(seq2[i], 0.0))
    
    def test_min_frames_filtering(self):
        """Test: Verificar que se descartan secuencias con muy pocos frames."""
        n_frames = 7
        min_frames = 3
        
        sequences, lengths, metas = group_spectrograms_to_sequences(
            self.dataset,
            n_frames=n_frames,
            min_frames=min_frames
        )
        
        # audio_3.wav tiene solo 2 frames, debería descartarse
        filenames = [m.filename for m in metas]
        self.assertNotIn('audio_3.wav', filenames)
        
        # Solo deberían quedar audio_1 y audio_2
        self.assertEqual(len(sequences), 2)


class TestLSTMModel(unittest.TestCase):
    """Tests para arquitectura del modelo LSTM-DA."""
    
    def setUp(self):
        """Setup: crear modelo de prueba."""
        self.n_frames = 7
        self.batch_size = 4
        self.lstm_units = 32
        self.n_domains = 4
        
        self.model = TimeCNNBiLSTM_DA(
            n_frames=self.n_frames,
            lstm_units=self.lstm_units,
            n_domains=self.n_domains,
            p_drop_conv=0.3,
            p_drop_fc=0.5,
        )
        self.model.eval()  # Modo evaluación para tests determinísticos
    
    def test_forward_output_shapes(self):
        """Test: Verificar dimensiones de outputs."""
        x = torch.randn(self.batch_size, self.n_frames, 1, 65, 41)
        
        logits_pd, logits_domain, _ = self.model(x, lengths=None)
        
        # Verificar shapes
        self.assertEqual(logits_pd.shape, (self.batch_size, 2))
        self.assertEqual(logits_domain.shape, (self.batch_size, self.n_domains))
    
    def test_forward_with_masking(self):
        """Test: Verificar que masking no cause errores."""
        x = torch.randn(self.batch_size, self.n_frames, 1, 65, 41)
        lengths = torch.tensor([7, 5, 6, 4])  # Diferentes longitudes
        
        logits_pd, logits_domain, _ = self.model(x, lengths=lengths)
        
        # Verificar que no hay NaNs
        self.assertFalse(torch.isnan(logits_pd).any())
        self.assertFalse(torch.isnan(logits_domain).any())
        
        # Verificar shapes
        self.assertEqual(logits_pd.shape, (self.batch_size, 2))
        self.assertEqual(logits_domain.shape, (self.batch_size, self.n_domains))
    
    def test_embeddings_return(self):
        """Test: Verificar que se retornan embeddings cuando se solicita."""
        x = torch.randn(self.batch_size, self.n_frames, 1, 65, 41)
        
        logits_pd, logits_domain, embeddings = self.model(
            x, lengths=None, return_embeddings=True
        )
        
        # Verificar que embeddings no es None
        self.assertIsNotNone(embeddings)
        
        # Verificar shape de embeddings (B, 2*lstm_units)
        expected_dim = 2 * self.lstm_units
        self.assertEqual(embeddings.shape, (self.batch_size, expected_dim))
    
    def test_deterministic_forward(self):
        """Test: Verificar que forward es determinístico en eval mode."""
        x = torch.randn(self.batch_size, self.n_frames, 1, 65, 41)
        
        # Dos forward passes con la misma entrada
        logits_pd_1, _, _ = self.model(x, lengths=None)
        logits_pd_2, _, _ = self.model(x, lengths=None)
        
        # Deberían ser idénticos en modo eval
        self.assertTrue(torch.allclose(logits_pd_1, logits_pd_2))


class TestGradientReversal(unittest.TestCase):
    """Tests para Gradient Reversal Layer."""
    
    def test_grl_forward(self):
        """Test: GRL no modifica valores en forward."""
        from modules.models.lstm_da.model import GradientReversalLayer
        
        grl = GradientReversalLayer(lambda_=1.0)
        x = torch.randn(4, 128, requires_grad=True)
        
        y = grl(x)
        
        # Forward debería ser identidad
        self.assertTrue(torch.allclose(x, y))
    
    def test_grl_backward(self):
        """Test: GRL invierte gradientes en backward."""
        from modules.models.lstm_da.model import GradientReversalLayer
        
        lambda_ = 1.0
        grl = GradientReversalLayer(lambda_=lambda_)
        
        # Input con gradientes
        x = torch.randn(4, 128, requires_grad=True)
        
        # Forward
        y = grl(x)
        
        # Backward con gradiente artificial
        grad_output = torch.ones_like(y)
        y.backward(grad_output)
        
        # Gradiente en x debería ser -lambda_ * grad_output
        expected_grad = -lambda_ * grad_output
        self.assertTrue(torch.allclose(x.grad, expected_grad))
    
    def test_lambda_update(self):
        """Test: Lambda se puede actualizar correctamente."""
        from modules.models.lstm_da.model import GradientReversalLayer
        
        grl = GradientReversalLayer(lambda_=0.0)
        self.assertEqual(grl.lambda_, 0.0)
        
        grl.set_lambda(0.5)
        self.assertEqual(grl.lambda_, 0.5)
        
        grl.set_lambda(1.0)
        self.assertEqual(grl.lambda_, 1.0)


class TestBiLSTMTemporal(unittest.TestCase):
    """Tests para procesamiento temporal con BiLSTM."""
    
    def setUp(self):
        """Setup: crear modelo simple."""
        self.model = TimeCNNBiLSTM_DA(
            n_frames=7,
            lstm_units=32,
            n_domains=4,
            p_drop_conv=0.0,  # Sin dropout para tests
            p_drop_fc=0.0,
        )
        self.model.eval()
    
    def test_sequence_processing(self):
        """Test: BiLSTM procesa secuencias correctamente."""
        batch_size = 2
        n_frames = 7
        
        # Crear dos secuencias idénticas excepto en un frame
        x1 = torch.randn(1, n_frames, 1, 65, 41)
        x2 = x1.clone()
        x2[0, 0, :, :, :] = torch.randn_like(x2[0, 0, :, :, :])  # Cambiar primer frame
        
        x = torch.cat([x1, x2], dim=0)  # Batch de 2
        
        logits_pd, _, _ = self.model(x)
        
        # Las predicciones deberían ser diferentes
        self.assertFalse(torch.allclose(logits_pd[0], logits_pd[1]))
    
    def test_masking_effect(self):
        """Test: Masking afecta el resultado correctamente."""
        batch_size = 2
        n_frames = 7
        
        # Secuencia con padding
        x = torch.randn(batch_size, n_frames, 1, 65, 41)
        
        # Forward sin masking (usa todos los frames)
        logits_no_mask, _, _ = self.model(x, lengths=None)
        
        # Forward con masking (usa solo primeros 4 frames)
        lengths = torch.tensor([4, 4])
        logits_with_mask, _, _ = self.model(x, lengths=lengths)
        
        # Los resultados deberían ser diferentes
        self.assertFalse(torch.allclose(logits_no_mask, logits_with_mask))


class TestIntegration(unittest.TestCase):
    """Tests de integración end-to-end."""
    
    def test_full_pipeline(self):
        """Test: Pipeline completo desde secuencias hasta predicciones."""
        # 1. Crear dataset mock
        dataset = []
        for i in range(8):
            spec = np.random.randn(65, 41).astype(np.float32)
            meta = type('obj', (object,), {
                'subject_id': 'patient_1',
                'filename': 'test.wav',
                'segment_id': i,
                'vowel_type': 'a',
                'condition': 'h',
            })()
            dataset.append({'spectrogram': spec, 'metadata': meta})
        
        # 2. Crear secuencias
        n_frames = 7
        sequences, lengths, metas = group_spectrograms_to_sequences(
            dataset, n_frames=n_frames, min_frames=3
        )
        
        self.assertEqual(len(sequences), 1)
        
        # 3. Convertir a tensors
        X = torch.from_numpy(np.stack(sequences, axis=0))  # (1, 7, 1, 65, 41)
        lengths_t = torch.tensor(lengths)
        
        # 4. Forward pass del modelo
        model = TimeCNNBiLSTM_DA(
            n_frames=n_frames,
            lstm_units=32,
            n_domains=4,
        )
        model.eval()
        
        with torch.no_grad():
            logits_pd, logits_domain, embeddings = model(
                X, lengths=lengths_t, return_embeddings=True
            )
        
        # 5. Verificar outputs
        self.assertEqual(logits_pd.shape, (1, 2))
        self.assertEqual(logits_domain.shape, (1, 4))
        self.assertEqual(embeddings.shape, (1, 64))  # 2*32
        
        # 6. Verificar que se pueden calcular probabilidades
        probs_pd = torch.softmax(logits_pd, dim=1)
        self.assertTrue(torch.allclose(probs_pd.sum(dim=1), torch.tensor([1.0])))


class TestParameterCount(unittest.TestCase):
    """Tests para verificar número de parámetros."""
    
    def test_parameter_count(self):
        """Test: Verificar que el modelo tiene parámetros razonables."""
        model = TimeCNNBiLSTM_DA(
            n_frames=7,
            lstm_units=64,
            n_domains=4,
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        
        # Verificar que tiene un número razonable de parámetros
        # Debería estar entre 500k y 2M
        self.assertGreater(total_params, 500_000)
        self.assertLess(total_params, 2_000_000)
        
        print(f"\nTotal parámetros: {total_params:,}")


def run_tests():
    """Ejecutar todos los tests."""
    print("=" * 70)
    print("PRUEBAS UNITARIAS: TIME-CNN-BILSTM-DA")
    print("=" * 70)
    print()
    
    # Crear test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Agregar tests
    suite.addTests(loader.loadTestsFromTestCase(TestSequenceGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestLSTMModel))
    suite.addTests(loader.loadTestsFromTestCase(TestGradientReversal))
    suite.addTests(loader.loadTestsFromTestCase(TestBiLSTMTemporal))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestParameterCount))
    
    # Ejecutar tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    print(f"Tests ejecutados: {result.testsRun}")
    print(f"Exitosos: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Fallidos: {len(result.failures)}")
    print(f"Errores: {len(result.errors)}")
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

