"""
Time-CNN-BiLSTM with Domain Adaptation Module
==============================================
Modelo Time-CNN-BiLSTM-DA para clasificación PD vs HC.
Diseñado para secuencias de espectrogramas Mel con Domain Adversarial Training.

Reference:
    Ibarra et al. (2023) "Towards a Corpus (and Language)-Independent
    Screening of Parkinson's Disease from Voice and Speech through
    Domain Adaptation"
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

# Importar componentes compartidos
from ..common.layers import (
    FeatureExtractor,
    GradientReversalLayer,
)


# ============================================================
# TIME-CNN-BILSTM WITH DOMAIN ADAPTATION
# ============================================================


class TimeCNNBiLSTM_DA(nn.Module):
    """
    Time-CNN-BiLSTM con Domain Adaptation para clasificación de Parkinson.

    Arquitectura según Ibarra et al. (2023):
        1. Time-distributed CNN 2D (extracción de features por frame)
        2. Projection layer (reducción dimensional)
        3. BiLSTM (modelado temporal)
        4. Global pooling temporal con masking
        5. Dual-head: PD detector + Domain detector (con GRL)

    Input shape: (B, T, 1, H, W) donde T = n_frames
    Output: (logits_pd, logits_domain)

    Notes:
        - A diferencia de CNN2D/CNN1D, no requiere post-proceso por paciente
        - La BiLSTM procesa toda la secuencia en un solo forward
        - El masking ignora frames de padding en la agregación temporal
    """

    def __init__(
        self,
        n_frames: int = 7,
        lstm_units: int = 64,
        n_domains: int = 4,
        p_drop_conv: float = 0.3,
        p_drop_fc: float = 0.5,
        input_shape: Tuple[int, int] = (65, 41),
    ):
        """
        Args:
            n_frames: Número de frames por secuencia (default: 7)
            lstm_units: Unidades en cada dirección de BiLSTM (default: 64)
            n_domains: Número de dominios/corpus (default: 4)
            p_drop_conv: Dropout en capas convolucionales
            p_drop_fc: Dropout en capas fully connected
            input_shape: Dimensiones de entrada por frame (H, W)
        """
        super().__init__()

        self.n_frames = n_frames
        self.lstm_units = lstm_units
        self.n_domains = n_domains
        self.p_drop_conv = p_drop_conv
        self.p_drop_fc = p_drop_fc
        self.input_shape = input_shape

        # Feature extractor time-distributed (reutilizado de CNN2D)
        self.feature_extractor = FeatureExtractor(
            p_drop_conv=p_drop_conv,
            input_shape=input_shape,
        )
        feature_dim = self.feature_extractor.feature_dim

        # Projection layer: reduce dimensión antes de LSTM
        self.emb_dim = 128
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, self.emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p_drop_fc),
        )

        # BiLSTM temporal
        self.lstm = nn.LSTM(
            input_size=self.emb_dim,
            hidden_size=lstm_units,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,  # Ya hay dropout en projection y heads
        )

        # Dimensión de features después de BiLSTM
        self.feat_dim = 2 * lstm_units  # bidirectional conquetación de las direcciones

        # PD Head (tarea principal)
        self.pd_head = nn.Sequential(
            nn.Linear(self.feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p_drop_fc),
            nn.Linear(64, 2),  # Binario: HC vs PD
        )

        # Gradient Reversal Layer
        self.grl = GradientReversalLayer(lambda_=0.0)

        # Domain Head (tarea auxiliar con GRL)
        self.domain_head = nn.Sequential(
            nn.Linear(self.feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p_drop_fc),
            nn.Linear(64, n_domains),
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass con time-distributed CNN, BiLSTM y dual-head.

        Args:
            x: Input tensor (B, T, 1, H, W)
            lengths: Longitudes reales de secuencias para masking (B,)
            return_embeddings: Si retornar embeddings LSTM para visualización

        Returns:
            logits_pd: Logits para clasificación PD (B, 2)
            logits_domain: Logits para clasificación de dominio (B, n_domains)
            embeddings: (opcional) Embeddings globales (B, feat_dim)
        """
        B, T, C, H, W = x.shape

        # Time-distributed CNN: procesar cada frame independientemente
        # Reshape: (B, T, C, H, W) → (B*T, C, H, W)
        x_flat = x.view(B * T, C, H, W)

        # Extraer features con CNN
        features = self.feature_extractor(x_flat)  # (B*T, 64, H', W')

        # Projection: (B*T, feat_dim) → (B*T, emb_dim)
        embeddings = self.projection(features)  # (B*T, emb_dim)

        # Reshape para LSTM: (B*T, emb_dim) → (B, T, emb_dim)
        embeddings = embeddings.view(B, T, -1)

        # Pack padded sequences si se proveen lengths
        if lengths is not None:
            # Asegurar que lengths esté en CPU para pack_padded_sequence
            lengths_cpu = lengths.cpu()
            embeddings = nn.utils.rnn.pack_padded_sequence(
                embeddings,
                lengths_cpu,
                batch_first=True,
                enforce_sorted=False,
            )

        # BiLSTM temporal
        lstm_out, _ = self.lstm(embeddings)  # (B, T, 2*lstm_units) o PackedSequence

        # Unpack si se usó packing
        if lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out,
                batch_first=True,
            )  # (B, T, 2*lstm_units)

        # Global pooling temporal con masking
        if lengths is None:
            # Sin masking: mean sobre todos los frames
            z = lstm_out.mean(dim=1)  # (B, 2*lstm_units)
        else:
            # Con masking: mean solo sobre frames válidos
            # Crear mask: (B, T)
            mask = torch.arange(lstm_out.size(1), device=lengths.device)[None, :] < lengths[:, None]
            mask = mask.unsqueeze(-1)  # (B, T, 1)

            # Mean pooling con mask
            z = (lstm_out * mask).sum(dim=1) / lengths.float().unsqueeze(1)  # (B, 2*lstm_units)

        # Dual heads
        logits_pd = self.pd_head(z)

        # Domain head con GRL
        z_reversed = self.grl(z)
        logits_domain = self.domain_head(z_reversed)

        if return_embeddings:
            return logits_pd, logits_domain, z
        return logits_pd, logits_domain, None

    def set_lambda(self, lambda_: float):
        """
        Actualiza el factor lambda de la GRL.

        Args:
            lambda_: Nuevo valor de lambda (0 a 1)
        """
        self.grl.set_lambda(lambda_)


# ============================================================
# UTILIDADES
# ============================================================


# Funciones utilitarias movidas a modules.models.common.layers


# ============================================================
# PRUEBA RÁPIDA
# ============================================================


if __name__ == "__main__":
    print("=" * 70)
    print("TEST: TIME-CNN-BILSTM-DA")
    print("=" * 70)

    # Parámetros
    n_frames = 7
    batch_size = 4
    lstm_units = 64
    n_domains = 4

    # Crear modelo
    model = TimeCNNBiLSTM_DA(
        n_frames=n_frames,
        lstm_units=lstm_units,
        n_domains=n_domains,
        p_drop_conv=0.3,
        p_drop_fc=0.5,
    )
    print_model_summary(model)

    # Test forward pass sin masking
    print("\n" + "=" * 70)
    print("TEST 1: FORWARD SIN MASKING")
    print("=" * 70)
    x = torch.randn(batch_size, n_frames, 1, 65, 41)
    print(f"Input shape: {tuple(x.shape)}")

    logits_pd, logits_domain, _ = model(x, lengths=None, return_embeddings=False)
    print(f"Output PD shape: {tuple(logits_pd.shape)} (2 clases: HC/PD)")
    print(f"Output Domain shape: {tuple(logits_domain.shape)} ({n_domains} dominios)")

    # Test forward pass con masking
    print("\n" + "=" * 70)
    print("TEST 2: FORWARD CON MASKING")
    print("=" * 70)
    lengths = torch.tensor([7, 5, 6, 4])  # Diferentes longitudes
    print(f"Lengths: {lengths.tolist()}")

    logits_pd2, logits_domain2, embeddings = model(
        x, lengths=lengths, return_embeddings=True
    )
    print(f"Output PD shape: {tuple(logits_pd2.shape)}")
    print(f"Output Domain shape: {tuple(logits_domain2.shape)}")
    print(f"Embeddings shape: {tuple(embeddings.shape)} (para t-SNE)")

    # Test lambda scheduling
    print("\n" + "=" * 70)
    print("TEST 3: GRADIENT REVERSAL LAYER")
    print("=" * 70)
    print(f"Lambda inicial: {model.grl.lambda_}")
    model.set_lambda(0.5)
    print(f"Lambda actualizado: {model.grl.lambda_}")

    # Forward con nuevo lambda
    logits_pd3, logits_domain3, _ = model(x, lengths=lengths)
    print("Forward pass con lambda=0.5 OK")

    print("\n[OK] Todos los tests pasaron correctamente")
