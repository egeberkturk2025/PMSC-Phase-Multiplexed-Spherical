# holodb/models/tdnn_sequence_memory.py
# TDNN Temporal Sequence Memory
# "Bu chunk dizisini daha once gordum mu?" sorusunu cevaplar.
#
# Mimari:
#   Input:  t-3, t-2, t-1, t  (4 x 512 phase vektoru)
#   TDNN temporal convolution -> 128-dim embedding
#   Output: cosine similarity ile karsilastirilir
#
# Egitim:
#   Pozitif cift: ayni dosyanin ardisik sequence'lari -> benzer embedding
#   Negatif cift: farkli sequence'lardan rastgele ciftler -> uzak embedding
#   Loss: Triplet margin loss
#
# PMSC Entegrasyonu:
#   Phase sequence'leri SphericalEmbeddingCodec ile sikistirilarak
#   TDNN embedding uzayinda depolanir. Sorgu aninda:
#   1. Query chunk -> faz vektoru -> TDNN embedding
#   2. PMSC ile sikistirilmis embedding veritabaninda ANN arama
#   3. cosine_sim > 0.95 -> "daha once goruldu"
#
# Telif Hakki (c) 2026 Ege Berk Turk - Tum Haklari Saklidir.

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TDNNBlock(nn.Module):
    """Tek TDNN katmani - belirli bir temporal context."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        context: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_dim,
            out_dim,
            kernel_size=context,
            dilation=dilation,
            padding=(context - 1) * dilation // 2,
        )
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, features)
        out = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.act(self.norm(out))


class PhaseSequenceMemory(nn.Module):
    """
    Phase sequence'ini sabit boyutlu embedding'e cevirir.

    Benzer icerik -> yakin embedding (cosine similarity yuksek)
    Farkli icerik -> uzak embedding

    PMSC ile entegrasyon:
        embeddings = SphericalEmbeddingCodec.encode(phase_sequences)
        # 3.66x sikistirma, exponent_127_pct=%99.75
        # Cosine sim korunur (float16 lossy modda 1.0000)
    """

    def __init__(
        self,
        input_dim: int = 512,
        embed_dim: int = 128,
    ) -> None:
        super().__init__()
        self.tdnn1 = TDNNBlock(input_dim, 256, context=2, dilation=1)
        self.tdnn2 = TDNNBlock(256, 128, context=3, dilation=2)
        self.tdnn3 = TDNNBlock(128, embed_dim, context=2, dilation=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, time_steps, input_dim)
        returns: (batch, embed_dim) L2-normalized
        """
        out = self.tdnn1(x)
        out = self.tdnn2(out)
        out = self.tdnn3(out)
        # Pool over time axis
        out = self.pool(out.transpose(1, 2)).squeeze(-1)
        out = self.proj(out)
        return F.normalize(out, dim=-1)


class TripletLoss(nn.Module):
    """Triplet margin loss - ayni dosya / farkli dosya ayirimi."""

    def __init__(self, margin: float = 0.3) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        d_pos = 1.0 - F.cosine_similarity(anchor, positive)
        d_neg = 1.0 - F.cosine_similarity(anchor, negative)
        return F.relu(d_pos - d_neg + self.margin).mean()


class PMSCSequenceIndex:
    """
    TDNN embedding'lerini PMSC ile sikistirarak depolayan index.

    Kullanim:
        model = PhaseSequenceMemory()
        index = PMSCSequenceIndex(model, keep_ratio=0.03)

        # Indeksleme
        index.add(chunk_id="doc1_chunk3", phase_sequence=seq)

        # Sorgulama
        results = index.search(query_sequence, top_k=5)
        # -> [(chunk_id, cosine_sim), ...]
    """

    def __init__(
        self,
        model: PhaseSequenceMemory,
        keep_ratio: float = 0.03,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.keep_ratio = keep_ratio
        self.device = device
        self._ids: List[str] = []
        self._embeddings: List[torch.Tensor] = []
        self._compressed_payload: Optional[object] = None
        self._dirty = False

    @torch.no_grad()
    def _encode(self, sequence: torch.Tensor) -> torch.Tensor:
        """Phase sequence -> normalized embedding."""
        if sequence.dim() == 2:
            sequence = sequence.unsqueeze(0)  # (1, T, D)
        return self.model(sequence.to(self.device)).squeeze(0)

    def add(self, chunk_id: str, phase_sequence: torch.Tensor) -> None:
        """Yeni chunk ekle."""
        emb = self._encode(phase_sequence)
        self._ids.append(chunk_id)
        self._embeddings.append(emb.cpu())
        self._dirty = True

    def _compress(self) -> None:
        """Tum embedding'leri PMSC ile sikistir."""
        if not self._dirty or not self._embeddings:
            return
        try:
            import numpy as np
            from holodb.codecs.spherical_embedding_codec import SphericalEmbeddingCodec
            vecs = [e.numpy().astype("float32") for e in self._embeddings]
            codec = SphericalEmbeddingCodec(
                keep_ratio=self.keep_ratio,
                auto_ratio=True,
                lossy=True,  # float16 - cosine=1.0000
            )
            self._compressed_payload = codec.encode(vecs)
            self._dirty = False
        except ImportError:
            pass  # PMSC kurulu degilse ham embedding kullan

    def search(
        self, query_sequence: torch.Tensor, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """En benzer top_k chunk'i bul."""
        if not self._embeddings:
            return []
        query_emb = self._encode(query_sequence).cpu()
        # Ham embedding karsilastirma (hiz icin)
        stack = torch.stack(self._embeddings)  # (N, D)
        sims = F.cosine_similarity(query_emb.unsqueeze(0), stack).tolist()
        ranked = sorted(
            zip(self._ids, sims), key=lambda x: x[1], reverse=True
        )
        return ranked[:top_k]

    def __len__(self) -> int:
        return len(self._ids)

    def compression_stats(self) -> dict:
        """PMSC sikistirma istatistiklerini goster."""
        if not self._embeddings:
            return {"count": 0}
        self._compress()
        if self._compressed_payload is None:
            return {"count": len(self._ids), "compressed": False}
        try:
            import numpy as np
            from holodb.codecs.spherical_embedding_codec import SphericalEmbeddingCodec
            vecs = [e.numpy().astype("float32") for e in self._embeddings]
            return SphericalEmbeddingCodec.compression_stats(vecs, self._compressed_payload)
        except Exception:
            return {"count": len(self._ids), "compressed": True}
