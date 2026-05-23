# holodb/core/seed_engine.py
# HoloDB SeedEngine - Ust Duzey Codec Yonlendirici ve API
# v0.5.0 - SphericalEmbeddingCodec + RGBMultiplexedCodec entegrasyonu
# Telif Hakki (c) 2026 Ege Berk Turk - Tum Haklari Saklidir.

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class EncodingResult:
    """Encode isleminin sonucu."""
    codec_type: str
    original_size: int
    encoded_size: int
    compression_ratio: float
    entropy: float
    encoding_time_ms: float
    seed_payload: Dict[str, Any]


@dataclass
class DecodingResult:
    """Decode isleminin sonucu."""
    codec_type: str
    decoded_size: int
    hash_verified: bool
    decoding_time_ms: float
    data: bytes


def _calculate_entropy(data: bytes) -> float:
    """Shannon entropy hesapla (bits/byte)."""
    if not data:
        return 0.0
    import math
    freq = {}
    for b in data:
        freq[b] = freq.get(b, 0) + 1
    n = len(data)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


class SeedEngine:
    """
    HoloDB merkezi encoding/decoding motoru.

    Desteklenen Codecler:
    1. DirectVoxel       - Ham piksel, bit-perfect
    2. GenerativeSeed    - Tohum tabanli uretim
    3. DCT               - Blok DCT sikistirma
    4. PhaseHologram     - Tek gorsel holografik
    5. MultiplexedHolographic - Faz coklama v0.5.0
    6. RGBMultiplexedCodec    - YCbCr kanal ayristirma
    7. EmbeddingCodec         - Float vektor 2D matris FFT
    8. SphericalEmbeddingCodec - Kuresel cokus + Faz coklama
    """

    LOW_ENTROPY_THRESHOLD = 4.0
    HIGH_ENTROPY_THRESHOLD = 7.5

    def __init__(self) -> None:
        self._lazy_codecs: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Entropy analizi
    # ------------------------------------------------------------------
    def analyze_entropy(self, data: bytes) -> Tuple[float, str]:
        entropy = _calculate_entropy(data)
        if entropy < self.LOW_ENTROPY_THRESHOLD:
            level = "low"
        elif entropy > self.HIGH_ENTROPY_THRESHOLD:
            level = "high"
        else:
            level = "medium"
        return entropy, level

    # ------------------------------------------------------------------
    # Deterministik adres turetme (HoloIndex ile uyumlu)
    # ------------------------------------------------------------------
    @staticmethod
    def derive_from_path(root_seed: int, path: str) -> int:
        """FNV-1a karisimi ile deterministik 31-bit seed turet."""
        FNV_PRIME = 0x01000193
        seed = root_seed & 0xFFFFFFFF
        norm = path.strip("/")
        if not norm:
            return seed & 0x7FFFFFFF
        for part in norm.split("/"):
            seed ^= 0xFF  # segment separator
            for byte in part.encode("utf-8"):
                seed = (seed ^ byte) * FNV_PRIME & 0xFFFFFFFF
        return seed & 0x7FFFFFFF

    # ------------------------------------------------------------------
    # Gorsel Batch API (v0.5.0)
    # ------------------------------------------------------------------
    def encode_image_batch(
        self,
        images: List[np.ndarray],
        keep_ratio: Optional[float] = None,
        lossy: bool = False,
        use_alpha: bool = True,
        differential: bool = False,
        lossy_only: bool = False,
    ) -> bytes:
        """
        N adet grayscale gorsel -> .holo binary blob.
        images: list of (H, W) uint8 numpy arrays
        """
        import tempfile, os
        from holodb.codecs.multiplexed_holographic import MultiplexedHolographicCodec

        codec = MultiplexedHolographicCodec(
            keep_ratio=keep_ratio or self.auto_keep_ratio(images),
            lossy=lossy,
        )
        if lossy_only:
            payload = codec.encode_lossy_only(images, use_alpha=use_alpha)
        else:
            payload = codec.encode_batch(
                images, use_alpha=use_alpha, differential=differential
            )
        import pickle
        return pickle.dumps(payload)

    def decode_image_batch(self, data: bytes, idx: int = 0) -> np.ndarray:
        """Binary blob'dan idx numarali gorseli decode et."""
        import pickle
        from holodb.codecs.multiplexed_holographic import MultiplexedHolographicCodec

        payload = pickle.loads(data)
        codec = MultiplexedHolographicCodec(
            keep_ratio=payload.keep_ratio, lossy=payload.lossy
        )
        return codec.decode_single(payload, idx)

    def auto_keep_ratio(self, images: List[np.ndarray]) -> float:
        """Faz korelasyonuna gore adaptif keep_ratio sec."""
        from holodb.codecs.multiplexed_holographic import MultiplexedHolographicCodec
        if len(images) < 2:
            return 0.02
        return MultiplexedHolographicCodec().auto_keep_ratio(images)

    # ------------------------------------------------------------------
    # RGB Batch API (v0.5.0)
    # ------------------------------------------------------------------
    def encode_rgb_batch(
        self,
        images: List[np.ndarray],
        keep_ratio_y: float = 0.02,
        keep_ratio_c: float = 0.005,
        lossy: bool = False,
        differential: bool = True,
    ) -> Dict[str, Any]:
        """
        N adet RGB gorsel (H,W,3) uint8 -> codec sonuc dict.
        Returns: {codec, payload, stats}
        """
        from holodb.codecs.rgb_holographic import RGBMultiplexedCodec

        codec = RGBMultiplexedCodec(
            keep_ratio_y=keep_ratio_y,
            keep_ratio_c=keep_ratio_c,
            use_alpha=True,
            differential=differential,
            auto_ratio=True,
        )
        if lossy:
            payload = codec.encode_rgb_lossy(images)
        else:
            payload = codec.encode_rgb_batch(images, lossy=False)
        stats = codec.compression_stats(images, payload)
        return {"codec": "rgb_multiplexed", "payload": payload, "stats": stats}

    def decode_rgb(self, result: Dict[str, Any], idx: int = 0) -> np.ndarray:
        """encode_rgb_batch ciktisinden tek RGB gorsel decode et."""
        from holodb.codecs.rgb_holographic import RGBMultiplexedCodec
        return RGBMultiplexedCodec().decode_rgb_single(result["payload"], idx)

    # ------------------------------------------------------------------
    # Embedding API (v0.5.0)
    # ------------------------------------------------------------------
    def encode_embeddings(
        self,
        vectors: List[np.ndarray],
        keep_ratio: float = 0.03,
        lossy: bool = False,
        differential: bool = False,
        use_spherical: bool = True,
    ) -> Dict[str, Any]:
        """
        N adet float embedding vektoru sikistir.
        use_spherical=True  -> SphericalEmbeddingCodec (3.66x, exp127=%99.75)
        use_spherical=False -> EmbeddingCodec          (2.86x, duz FFT)
        Returns: {codec, payload, stats}
        """
        if use_spherical:
            from holodb.codecs.spherical_embedding_codec import SphericalEmbeddingCodec
            codec = SphericalEmbeddingCodec(
                keep_ratio=keep_ratio,
                use_alpha=True,
                differential=differential,
                auto_ratio=True,
                lossy=lossy,
            )
            payload = codec.encode(vectors)
            stats = SphericalEmbeddingCodec.compression_stats(vectors, payload)
            return {"codec": "spherical_embedding", "payload": payload, "stats": stats}
        else:
            from holodb.codecs.embedding_holographic import EmbeddingCodec
            codec = EmbeddingCodec(
                keep_ratio=keep_ratio,
                use_alpha=True,
                differential=differential,
                auto_ratio=True,
                lossy=lossy,
            )
            payload = codec.encode(vectors)
            stats = EmbeddingCodec.compression_stats(vectors, payload)
            return {"codec": "embedding", "payload": payload, "stats": stats}

    def decode_embedding(
        self, result: Dict[str, Any], idx: int = 0
    ) -> np.ndarray:
        """encode_embeddings ciktisinden tek vektor decode et."""
        codec_name = result["codec"]
        payload = result["payload"]
        if codec_name == "spherical_embedding":
            from holodb.codecs.spherical_embedding_codec import SphericalEmbeddingCodec
            return SphericalEmbeddingCodec.decode(payload, idx)
        else:
            from holodb.codecs.embedding_holographic import EmbeddingCodec
            return EmbeddingCodec.decode(payload, idx)

    # ------------------------------------------------------------------
    # Istatistik
    # ------------------------------------------------------------------
    def get_stats(self) -> Dict[str, Any]:
        return {
            "codecs_available": [
                "generative_seed", "direct_voxel", "spherical",
                "multiplexed_holographic", "rgb_multiplexed",
                "embedding_codec", "spherical_embedding",
            ],
            "entropy_thresholds": {
                "low": self.LOW_ENTROPY_THRESHOLD,
                "high": self.HIGH_ENTROPY_THRESHOLD,
            },
            "principle": "Veri Saklanmaz, Uretilir",
            "version": "0.5.0",
        }
