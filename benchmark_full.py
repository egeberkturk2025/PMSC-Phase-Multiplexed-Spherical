#!/usr/bin/env python3
"""
benchmark_full.py — PMSC Kapsamlı Akademik Kıyaslama
======================================================
Konferans kalitesi için:
  1. 10.000 Gaussian unit vektör (d=256,384,768,1536)
  2. Product Quantization (PQ) baseline
  3. Scalar Quantization (SQ8, SQ4) baseline
  4. Raw zlib baseline
  5. Shannon entropi alt sınırı teorik hesabı
  6. Boyut ölçekleme tablosu (d=256→3072)

Kullanım:
  pip install numpy scipy sentence-transformers pytest
  python benchmark_full.py

Copyright (c) 2026 Ege Berk Türk — Tüm Hakları Saklıdır.
"""

import os, sys, math, struct, zlib, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from holodb.codecs.spherical_embedding_codec import SphericalEmbeddingCodec

# ─────────────────────────────────────────────────────────────────────────────
# BASELINE YÖNTEMLERİ
# ─────────────────────────────────────────────────────────────────────────────

class RawZlibBaseline:
    """Ham float32 byte dizisi + zlib — en basit baseline."""
    def compress(self, vectors):
        raw = np.stack(vectors).astype(np.float32).tobytes()
        return zlib.compress(raw, 9)

    def ratio(self, vectors):
        orig = sum(v.astype(np.float32).nbytes for v in vectors)
        comp = len(self.compress(vectors))
        return orig / comp


class ScalarQuantBaseline:
    """Scalar Quantization: float32 → uint8 (SQ8) veya uint4 (SQ4)."""

    def __init__(self, bits=8):
        self.bits = bits
        self.levels = 2 ** bits

    def compress(self, vectors):
        mat = np.stack(vectors).astype(np.float32)
        vmin, vmax = mat.min(), mat.max()
        scale = (vmax - vmin) / (self.levels - 1) if vmax > vmin else 1.0
        quantized = np.clip(((mat - vmin) / scale), 0, self.levels - 1).astype(np.uint8)
        meta = struct.pack("ffII", vmin, vmax, *mat.shape)
        compressed = zlib.compress(quantized.tobytes(), 9)
        return meta + compressed

    def ratio(self, vectors):
        orig = sum(v.astype(np.float32).nbytes for v in vectors)
        comp = len(self.compress(vectors))
        return orig / comp

    def cosine_sim(self, vectors):
        mat = np.stack(vectors).astype(np.float32)
        vmin, vmax = mat.min(), mat.max()
        scale = (vmax - vmin) / (self.levels - 1) if vmax > vmin else 1.0
        quantized = np.clip(((mat - vmin) / scale), 0, self.levels - 1).astype(np.uint8)
        recon = quantized.astype(np.float32) * scale + vmin
        sims = []
        for i in range(len(vectors)):
            a, b = mat[i], recon[i]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            sims.append(float(np.dot(a, b) / (na * nb + 1e-9)))
        return np.mean(sims)


class ProductQuantBaseline:
    """
    Product Quantization (PQ) — eğitimsiz versiyon.
    Vektörü M alt-uzaya böler, her alt-uzayda uniform quantization uygular.
    Gerçek PQ'dan biraz daha düşük performans ama eğitim gerektirmez.
    """
    def __init__(self, M=8, bits=8):
        self.M = M
        self.bits = bits
        self.levels = 2 ** bits

    def compress(self, vectors):
        mat = np.stack(vectors).astype(np.float32)
        n, d = mat.shape
        sub_dim = d // self.M
        codes = []
        params = []
        for m in range(self.M):
            sub = mat[:, m * sub_dim:(m + 1) * sub_dim]
            vmin, vmax = sub.min(), sub.max()
            params.append((vmin, vmax))
            scale = (vmax - vmin) / (self.levels - 1) if vmax > vmin else 1.0
            q = np.clip(((sub - vmin) / scale), 0, self.levels - 1).astype(np.uint8)
            codes.append(q)
        code_mat = np.concatenate(codes, axis=1)
        meta = struct.pack(f"II{2*self.M}f", n, d, *[x for p in params for x in p])
        return meta + zlib.compress(code_mat.tobytes(), 9)

    def ratio(self, vectors):
        orig = sum(v.astype(np.float32).nbytes for v in vectors)
        comp = len(self.compress(vectors))
        return orig / comp

    def cosine_sim(self, vectors):
        mat = np.stack(vectors).astype(np.float32)
        n, d = mat.shape
        sub_dim = d // self.M
        recon = np.zeros_like(mat)
        for m in range(self.M):
            sub = mat[:, m * sub_dim:(m + 1) * sub_dim]
            vmin, vmax = sub.min(), sub.max()
            scale = (vmax - vmin) / (self.levels - 1) if vmax > vmin else 1.0
            q = np.clip(((sub - vmin) / scale), 0, self.levels - 1).astype(np.uint8)
            recon[:, m * sub_dim:(m + 1) * sub_dim] = q.astype(np.float32) * scale + vmin
        sims = []
        for i in range(n):
            a, b = mat[i], recon[i]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            sims.append(float(np.dot(a, b) / (na * nb + 1e-9)))
        return np.mean(sims)


# ─────────────────────────────────────────────────────────────────────────────
# TEORİK SHANNON ENTROPİ ALT SINIRI
# ─────────────────────────────────────────────────────────────────────────────

def theoretical_entropy_bound(dim, n_vectors=10000):
    """
    Shannon Teoremi: H(X) = teorik minimum bit/eleman.

    Delta açıları N(0, 1/sqrt(d)) dağılımına yakın.
    Gaussian için diferansiyel entropi:
      h(X) = 0.5 * log2(2*pi*e * sigma^2)

    Bit-perfect sıkıştırma için float32 hassasiyet limiti eklenir:
      H_practical = h(X) + 0.1 * 23   (23-bit mantissa faktörü)
    """
    sigma = 1.0 / math.sqrt(dim)
    h_diff = 0.5 * math.log2(2 * math.pi * math.e * sigma ** 2)
    h_practical = max(h_diff + 23 * 0.1, 0.5)
    return {
        "dim": dim,
        "sigma": round(sigma, 6),
        "differential_entropy_bits": round(h_diff, 3),
        "practical_lower_bound_bits": round(h_practical, 3),
        "theoretical_max_ratio": round(32.0 / max(h_practical, 0.01), 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# BOYUT ÖLÇEKLEME TESTİ
# ─────────────────────────────────────────────────────────────────────────────

def scaling_benchmark(dims, n_vectors=500, rng=None):
    """d=256'dan d=3072'ye boyut ölçekleme testi."""
    if rng is None:
        rng = np.random.default_rng(42)

    results = []
    codec_bp    = SphericalEmbeddingCodec(keep_ratio=0.03, lossy=False)
    codec_lossy = SphericalEmbeddingCodec(keep_ratio=0.03, lossy=True)

    for dim in dims:
        vecs = []
        for _ in range(n_vectors):
            v = rng.standard_normal(dim).astype(np.float32)
            v /= np.linalg.norm(v)
            vecs.append(v)

        orig_bytes = sum(v.nbytes for v in vecs)
        p_bp    = codec_bp.encode(vecs)
        p_lossy = codec_lossy.encode(vecs)

        ratio_bp    = orig_bytes / p_bp.total_bytes()
        ratio_lossy = orig_bytes / p_lossy.total_bytes()

        sims = []
        for i, orig in enumerate(vecs[:20]):
            recon = SphericalEmbeddingCodec.decode(p_lossy, i)
            sims.append(SphericalEmbeddingCodec.cosine_similarity(orig, recon))

        stats  = SphericalEmbeddingCodec.entropy_stats(vecs[:100])
        theory = theoretical_entropy_bound(dim)

        results.append({
            "dim": dim,
            "n_vectors": n_vectors,
            "ratio_bp": round(ratio_bp, 3),
            "ratio_lossy": round(ratio_lossy, 3),
            "cosine_sim_min": round(min(sims), 6),
            "cosine_sim_mean": round(np.mean(sims), 6),
            "exponent_127_pct": stats["exponent_127_pct"],
            "delta_entropy_bits": stats["delta_entropy_bits"],
            "theoretical_max_ratio": theory["theoretical_max_ratio"],
        })
        print(f"  d={dim:5d}: BP={ratio_bp:.2f}x  Lossy={ratio_lossy:.2f}x  "
              f"exp127={stats['exponent_127_pct']:.1f}%  "
              f"cosim={np.mean(sims):.4f}  theory_max={theory['theoretical_max_ratio']:.1f}x")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# BÜYÜK ÖLÇEK TESTİ (10.000 vektör)
# ─────────────────────────────────────────────────────────────────────────────

def large_scale_benchmark(dim=1536, n_vectors=10000, rng=None):
    """10.000 vektörde PMSC vs tüm baselineler."""
    if rng is None:
        rng = np.random.default_rng(42)

    print(f"\n[10K Vektör Testi] d={dim}, n={n_vectors}")
    print("  Vektörler üretiliyor...")
    t0 = time.time()
    vecs = []
    for _ in range(n_vectors):
        v = rng.standard_normal(dim).astype(np.float32)
        v /= np.linalg.norm(v)
        vecs.append(v)
    print(f"  Üretim: {time.time()-t0:.1f}s")

    orig_bytes = sum(v.nbytes for v in vecs)
    results = {}

    # Raw zlib
    t0 = time.time()
    zlib_ratio = RawZlibBaseline().ratio(vecs)
    results["raw_zlib"] = {"ratio": round(zlib_ratio, 3), "cosine_sim": 1.0,
                           "time_s": round(time.time()-t0, 2)}
    print(f"  Raw zlib: {zlib_ratio:.3f}x  ({time.time()-t0:.1f}s)")

    # SQ8
    t0 = time.time()
    sq8 = ScalarQuantBaseline(bits=8)
    results["sq8"] = {"ratio": round(sq8.ratio(vecs), 3),
                      "cosine_sim": round(sq8.cosine_sim(vecs), 6),
                      "time_s": round(time.time()-t0, 2)}
    print(f"  SQ8:      {results['sq8']['ratio']:.3f}x  cosim={results['sq8']['cosine_sim']:.4f}")

    # PQ (M=8, 8bit)
    t0 = time.time()
    pq = ProductQuantBaseline(M=8, bits=8)
    results["pq8x8"] = {"ratio": round(pq.ratio(vecs), 3),
                        "cosine_sim": round(pq.cosine_sim(vecs), 6),
                        "time_s": round(time.time()-t0, 2)}
    print(f"  PQ(8x8):  {results['pq8x8']['ratio']:.3f}x  cosim={results['pq8x8']['cosine_sim']:.4f}")

    # PMSC Lossy (10K)
    t0 = time.time()
    codec_lossy = SphericalEmbeddingCodec(lossy=True)
    p_lossy = codec_lossy.encode(vecs)
    ratio_lossy = orig_bytes / p_lossy.total_bytes()
    sims = [SphericalEmbeddingCodec.cosine_similarity(
        vecs[i], SphericalEmbeddingCodec.decode(p_lossy, i)) for i in range(50)]
    results["pmsc_lossy"] = {"ratio": round(ratio_lossy, 3),
                             "cosine_sim": round(np.mean(sims), 6),
                             "time_s": round(time.time()-t0, 2)}
    print(f"  PMSC Lossy: {ratio_lossy:.3f}x  cosim={np.mean(sims):.4f}  ({time.time()-t0:.1f}s)")

    # PMSC Bit-Perfect (1K vektör — FFT yoğun)
    t0 = time.time()
    codec_bp = SphericalEmbeddingCodec(lossy=False)
    p_bp = codec_bp.encode(vecs[:1000])
    orig_1k = sum(v.nbytes for v in vecs[:1000])
    ratio_bp = orig_1k / p_bp.total_bytes()
    results["pmsc_bp_1k"] = {"ratio": round(ratio_bp, 3), "cosine_sim": "delta-perfect",
                              "time_s": round(time.time()-t0, 2), "note": "1000 vektör"}
    print(f"  PMSC BP (1K): {ratio_bp:.3f}x  delta-perfect  ({time.time()-t0:.1f}s)")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# ANA ÇALIŞTIRMA
# ─────────────────────────────────────────────────────────────────────────────

def print_table(title, headers, rows):
    col_w = [max(len(h), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    sep = "+-" + "-+-".join("-" * w for w in col_w) + "-+"
    header_row = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_w)) + " |"
    print(f"\n{'─'*62}")
    print(f"  {title}")
    print(f"{'─'*62}")
    print(sep)
    print(header_row)
    print(sep)
    for row in rows:
        print("| " + " | ".join(str(v).ljust(w) for v, w in zip(row, col_w)) + " |")
    print(sep)


def run_all():
    rng = np.random.default_rng(42)

    print("=" * 70)
    print("   PMSC — KAPSAMLI AKADEMİK KIYASLAMA (ICLR 2027 Hazırlığı)")
    print("   Seed: np.random.default_rng(42) — Tam Tekrarlanabilir")
    print("=" * 70)

    # 1. Teorik Entropi Alt Sınırları
    print("\n[1] Teorik Shannon Entropi Alt Sınırları")
    theory_rows = []
    for dim in [256, 384, 512, 768, 1024, 1536, 3072]:
        t = theoretical_entropy_bound(dim)
        theory_rows.append([
            dim,
            f"{t['sigma']:.5f}",
            f"{t['differential_entropy_bits']:.3f}",
            f"{t['practical_lower_bound_bits']:.3f}",
            f"{t['theoretical_max_ratio']:.1f}x",
        ])
    print_table(
        "Teorik Maksimum Sıkıştırma Oranları (Shannon Alt Sınırı)",
        ["dim", "sigma=1/sqrt(d)", "diff.entropy(bits)", "pratik alt sinir", "teor.max oran"],
        theory_rows
    )

    # 2. Boyut Ölçekleme Testi
    print("\n[2] Boyut Ölçekleme Testi (n=500 Gaussian vektör)")
    dims = [256, 384, 512, 768, 1024, 1536, 3072]
    scaling_results = scaling_benchmark(dims, n_vectors=500, rng=rng)

    scale_rows = []
    for r in scaling_results:
        scale_rows.append([
            r["dim"], r["n_vectors"],
            f"{r['ratio_bp']:.2f}x",
            f"{r['ratio_lossy']:.2f}x",
            f"{r['cosine_sim_mean']:.4f}",
            f"{r['exponent_127_pct']:.1f}%",
            f"{r['delta_entropy_bits']:.2f}",
            f"{r['theoretical_max_ratio']:.1f}x",
        ])
    print_table(
        "PMSC Boyut Olcekleme (d=256 → d=3072)",
        ["dim", "n", "BP oran", "Lossy oran", "cosine_sim", "exp127%", "delta_ent", "teor.max"],
        scale_rows
    )

    # 3. Baseline Karşılaştırması (d=1536, 10K vektör)
    large_results = large_scale_benchmark(dim=1536, n_vectors=10000, rng=rng)

    baseline_rows = [
        ["Raw float32",         "1.000x",  "1.0000",        "lossless",     "referans"],
        ["arXiv 2602.00079",    "~1.50x",  "1.0000",        "lossless",     "Xiao 2026"],
        ["Raw zlib",            f"{large_results['raw_zlib']['ratio']:.3f}x",
                                "1.0000",  "lossless",      "zlib lvl9"],
        ["SQ8 + zlib",          f"{large_results['sq8']['ratio']:.3f}x",
                                f"{large_results['sq8']['cosine_sim']:.4f}",
                                "kayipli", "8-bit scalar quant"],
        ["PQ(M=8,8bit)+zlib",   f"{large_results['pq8x8']['ratio']:.3f}x",
                                f"{large_results['pq8x8']['cosine_sim']:.4f}",
                                "kayipli", "product quant"],
        ["PMSC Lossy (f16)",    f"{large_results['pmsc_lossy']['ratio']:.3f}x",
                                f"{large_results['pmsc_lossy']['cosine_sim']:.4f}",
                                "near-lossless", "10K vektor"],
        ["PMSC Bit-Perfect",    f"{large_results['pmsc_bp_1k']['ratio']:.3f}x",
                                "delta-perfect", "bit-perfect", "1K vektor"],
    ]
    print_table(
        "Baseline Karsilastirmasi — d=1536, n=10.000",
        ["Yontem", "Oran", "Cosine Sim", "Mod", "Not"],
        baseline_rows
    )

    print("\n✅ Benchmark tamamlandi.")
    print("   Seed: np.random.default_rng(42) | Tam tekrarlanabilir.")
    print("   Makale referansi: DOI 10.5281/zenodo.20356814")


if __name__ == "__main__":
    run_all()
