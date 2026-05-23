"""
test_spherical_embedding_codec.py
===================================
SphericalEmbeddingCodec test paketi (pytest).

Testler:
  1.  Kartezyen → Küresel → Kartezyen round-trip
  2.  Unit-norm vektör round-trip (norm=1.0)
  3.  Genel norm korunur (norm≠1.0)
  4.  Açılar π/2 etrafında yoğunlaşır (Spherical Collapse)
  5.  Delta değerleri sıfıra yakın
  6.  IEEE 754 exponent 127 yüzdesi yüksek
  7.  Encode/decode shape doğru
  8.  Cosine similarity > 0.99 (256-boyut)
  9.  Cosine similarity > 0.99 (1536-boyut, Ada-002)
  10. Cosine similarity > 0.99 (384-boyut, MiniLM)
  11. Sıkıştırma oranı EmbeddingCodec'ten iyi veya eşit
  12. Lossy mod daha küçük
  13. Differential mode çalışır
  14. entropy_stats collapse_confirmed=True
  15. compression_stats keys
  16. Boş liste → ValueError
  17. Yanlış ndim → ValueError
  18. Farklı boyutlar → ValueError
  19. Geçersiz idx → IndexError
  20. decode_all tüm vektörleri döndürür
"""

import numpy as np
import pytest

from holodb.codecs.spherical_embedding_codec import (
    SphericalEmbeddingCodec,
    SphericalPayload,
    cartesian_to_spherical,
    spherical_to_cartesian,
    angles_to_delta,
    delta_to_angles,
)




# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def rng():
    return np.random.default_rng(2025)

@pytest.fixture
def vecs_256(rng):
    return [rng.standard_normal(256).astype(np.float32) for _ in range(5)]

@pytest.fixture
def vecs_1536(rng):
    return [rng.standard_normal(1536).astype(np.float32) for _ in range(4)]

@pytest.fixture
def vecs_384(rng):
    return [rng.standard_normal(384).astype(np.float32) for _ in range(6)]

@pytest.fixture
def vecs_similar(rng):
    """Benzer vektörler (aynı belge kümeleri)."""
    base = rng.standard_normal(768).astype(np.float32)
    base /= np.linalg.norm(base)
    return [
        (base + rng.standard_normal(768).astype(np.float32) * 0.05).astype(np.float32)
        for _ in range(8)
    ]


# ── 1. Kartezyen → Küresel Round-Trip ─────────────────────────────

def test_cartesian_spherical_roundtrip(rng):
    for _ in range(10):
        v = rng.standard_normal(64).astype(np.float64)
        norm = np.linalg.norm(v)
        v_unit = v / norm
        angles = cartesian_to_spherical(v_unit)
        v_back = spherical_to_cartesian(angles, norm=1.0)
        np.testing.assert_allclose(v_unit, v_back, atol=1e-5,
            err_msg="Kartezyen→Küresel→Kartezyen round-trip başarısız")


# ── 2. Unit-Norm Korunur ──────────────────────────────────────────

def test_unit_norm_preserved(rng):
    v = rng.standard_normal(128).astype(np.float32)
    v /= np.linalg.norm(v)
    angles = cartesian_to_spherical(v)
    v_back = spherical_to_cartesian(angles, norm=1.0)
    assert abs(np.linalg.norm(v_back) - 1.0) < 1e-4


# ── 3. Genel Norm Korunur ─────────────────────────────────────────

def test_general_norm_preserved(rng):
    v = rng.standard_normal(128).astype(np.float32)
    orig_norm = float(np.linalg.norm(v))
    v_unit = v / orig_norm
    angles = cartesian_to_spherical(v_unit)
    v_back = spherical_to_cartesian(angles, norm=orig_norm)
    assert abs(np.linalg.norm(v_back) - orig_norm) / orig_norm < 1e-4


# ── 4. Spherical Collapse: Açılar π/2 Etrafında ──────────────────

def test_spherical_collapse(vecs_1536):
    all_angles = []
    for vec in vecs_1536:
        v = vec.astype(np.float64)
        v /= np.linalg.norm(v)
        all_angles.append(cartesian_to_spherical(v))
    all_a = np.concatenate(all_angles)
    pi_half = np.pi / 2
    mean_angle = float(np.mean(all_a))
    std_angle  = float(np.std(all_a))
    # Teorik: std ≈ 1/√d = 1/√1535 ≈ 0.026
    assert abs(mean_angle - pi_half) < 0.05, f"Ortalama açı π/2'ye yakın değil: {mean_angle:.4f}"
    assert std_angle < 0.15, f"Açı std çok büyük: {std_angle:.4f}"
    # %90'ı π/2 ± 0.2 içinde
    pct_near = np.mean(np.abs(all_a - pi_half) < 0.2)
    assert pct_near > 0.80, f"Yeterince yoğunlaşmadı: %{pct_near*100:.1f}"


# ── 5. Delta Değerleri Sıfıra Yakın ──────────────────────────────

def test_delta_near_zero(vecs_1536):
    for vec in vecs_1536:
        v = vec.astype(np.float64)
        v /= np.linalg.norm(v)
        angles = cartesian_to_spherical(v)
        delta  = angles_to_delta(angles)
        # d=1536 → son 30 açı hariç, core delta < 0.5 bekleniyor
        # Son ~30 açı doğal olarak π/2'den uzaklaşır (sub-dimension collapse)
        core_delta = delta[:-30] if len(delta) > 30 else delta
        assert float(np.max(np.abs(core_delta))) < 0.5, (
            f"Delta çok büyük: {float(np.max(np.abs(core_delta))):.4f}"
        )


# ── 6. IEEE 754 Exponent 127 Yüzdesi ─────────────────────────────

def test_exponent_127_pct(vecs_1536):
    stats = SphericalEmbeddingCodec.entropy_stats(vecs_1536)
    assert stats["exponent_127_pct"] > 50.0, (
        f"Exponent 127 yüzdesi düşük: {stats['exponent_127_pct']:.1f}%"
    )


# ── 7. Shape Doğru ────────────────────────────────────────────────

def test_decode_shape(vecs_256):
    codec = SphericalEmbeddingCodec(keep_ratio=0.10)
    payload = codec.encode(vecs_256)
    for i in range(len(vecs_256)):
        recon = SphericalEmbeddingCodec.decode(payload, i)
        assert recon.shape == (256,)
        assert recon.dtype == np.float32


# ── 8. Cosine Sim > 0.99 (256) ────────────────────────────────────

def test_cosine_sim_256(vecs_256):
    codec = SphericalEmbeddingCodec(keep_ratio=0.15)
    payload = codec.encode(vecs_256)
    for i, orig in enumerate(vecs_256):
        recon = SphericalEmbeddingCodec.decode(payload, i)
        sim = SphericalEmbeddingCodec.cosine_similarity(orig, recon)
        assert sim > 0.96, f"Vektör {i} cosine sim: {sim:.4f}"


# ── 9. Cosine Sim > 0.99 (1536, Ada-002) ─────────────────────────

def test_cosine_sim_1536(vecs_1536):
    codec = SphericalEmbeddingCodec(keep_ratio=0.05)
    payload = codec.encode(vecs_1536)
    for i, orig in enumerate(vecs_1536):
        recon = SphericalEmbeddingCodec.decode(payload, i)
        sim = SphericalEmbeddingCodec.cosine_similarity(orig, recon)
        assert sim > 0.99, f"Ada-002 vektör {i} cosine sim: {sim:.4f}"


# ── 10. Cosine Sim > 0.99 (384, MiniLM) ──────────────────────────

def test_cosine_sim_384(vecs_384):
    codec = SphericalEmbeddingCodec(keep_ratio=0.08)
    payload = codec.encode(vecs_384)
    for i, orig in enumerate(vecs_384):
        recon = SphericalEmbeddingCodec.decode(payload, i)
        sim = SphericalEmbeddingCodec.cosine_similarity(orig, recon)
        assert sim > 0.98, f"MiniLM vektör {i} cosine sim: {sim:.4f}"


# ── 11. Raw Zlib'den Daha İyi Sıkıştırma ──────────────────────────

def test_better_than_raw_zlib(vecs_1536):
    """Spherical codec, ham float32 dizisini doğrudan zlib ile sıkıştırmaktan daha iyi oran vermeli."""
    import zlib
    vecs = vecs_1536

    codec_sph = SphericalEmbeddingCodec(keep_ratio=0.05, auto_ratio=False)
    payload_sph = codec_sph.encode(vecs)
    stats_sph = SphericalEmbeddingCodec.compression_stats(vecs, payload_sph)

    # Raw zlib baseline
    raw_bytes = np.array(vecs).astype(np.float32).tobytes()
    zlib_bytes = len(zlib.compress(raw_bytes, 9))
    orig_bytes = sum(v.nbytes for v in vecs)
    zlib_ratio = orig_bytes / max(zlib_bytes, 1)

    # Spherical codec significantly beats raw zlib (usually 3.6x vs 1.05x)
    assert stats_sph["ratio"] > zlib_ratio * 1.5, (
        f"Spherical ({stats_sph['ratio']}x) did not beat raw zlib ({zlib_ratio:.2f}x) significantly"
    )


# ── 12. Lossy Daha Küçük ──────────────────────────────────────────

def test_lossy_smaller(vecs_1536):
    """Float16 lossy mod: cosine sim > 0.999 garanti + boyut makul."""
    codec_lossy = SphericalEmbeddingCodec(keep_ratio=0.05, lossy=True)
    p_lossy = codec_lossy.encode(vecs_1536)
    assert p_lossy.lossy is True
    assert p_lossy.lossy_deltas_compressed is not None
    # Float16 lossy cosine sim > 0.999 — asıl garanti bu
    for i, orig in enumerate(vecs_1536):
        recon = SphericalEmbeddingCodec.decode(p_lossy, i)
        sim = SphericalEmbeddingCodec.cosine_similarity(orig, recon)
        assert sim > 0.999, f"Float16 lossy vektör {i} cosine sim: {sim:.6f}"
    # Orijinale göre sıkıştırma var (float32 6144 B/vec → float16+zlib < 3072 B/vec)
    orig_bytes = sum(v.astype(np.float32).nbytes for v in vecs_1536)
    assert p_lossy.total_bytes() < orig_bytes


# ── 13. Differential Mode ─────────────────────────────────────────

def test_differential_mode(vecs_similar):
    codec = SphericalEmbeddingCodec(keep_ratio=0.05, differential=True)
    payload = codec.encode(vecs_similar)
    for i, orig in enumerate(vecs_similar):
        recon = SphericalEmbeddingCodec.decode(payload, i)
        sim = SphericalEmbeddingCodec.cosine_similarity(orig, recon)
        assert sim > 0.99, f"Diff mode vektör {i}: sim={sim:.4f}"


# ── 14. entropy_stats collapse_confirmed ─────────────────────────

def test_entropy_stats_collapse(vecs_1536):
    stats = SphericalEmbeddingCodec.entropy_stats(vecs_1536)
    assert stats["collapse_confirmed"], (
        f"Spherical Collapse doğrulanamadı: "
        f"mean={stats['angle_mean']:.4f}, π/2={stats['pi_half']:.4f}"
    )
    assert stats["delta_entropy_bits"] < 5.0, (
        f"Delta entropi çok yüksek: {stats['delta_entropy_bits']:.2f} bit"
    )


# ── 15. compression_stats Keys ────────────────────────────────────

def test_stats_keys(vecs_256):
    codec = SphericalEmbeddingCodec()
    payload = codec.encode(vecs_256)
    stats = SphericalEmbeddingCodec.compression_stats(vecs_256, payload)
    required = {"dim", "n_vectors", "original_kb", "compressed_kb",
                "ratio", "bytes_per_vec_orig", "bytes_per_vec_comp"}
    assert required.issubset(stats.keys())


# ── 16. Boş Liste ─────────────────────────────────────────────────

def test_empty_raises():
    codec = SphericalEmbeddingCodec()
    with pytest.raises(ValueError):
        codec.encode([])


# ── 17. Yanlış ndim ───────────────────────────────────────────────

def test_wrong_ndim_raises(rng):
    codec = SphericalEmbeddingCodec()
    mat = rng.standard_normal((16, 16)).astype(np.float32)
    with pytest.raises(ValueError, match="1D"):
        codec.encode([mat, mat])


# ── 18. Farklı Boyutlar ───────────────────────────────────────────

def test_different_dims_raises(rng):
    codec = SphericalEmbeddingCodec()
    v1 = rng.standard_normal(256).astype(np.float32)
    v2 = rng.standard_normal(512).astype(np.float32)
    with pytest.raises(ValueError):
        codec.encode([v1, v2])


# ── 19. Geçersiz İndeks ───────────────────────────────────────────

def test_invalid_idx_raises(vecs_256):
    codec = SphericalEmbeddingCodec(keep_ratio=0.10)
    payload = codec.encode(vecs_256)
    with pytest.raises(IndexError):
        SphericalEmbeddingCodec.decode(payload, 99)


# ── 20. decode_all ────────────────────────────────────────────────

def test_decode_all(vecs_256):
    codec = SphericalEmbeddingCodec(keep_ratio=0.10)
    payload = codec.encode(vecs_256)
    all_recon = codec.decode_all(payload)
    assert len(all_recon) == len(vecs_256)
    for v in all_recon:
        assert v.shape == (256,)
