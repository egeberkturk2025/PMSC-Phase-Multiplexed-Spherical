"""
PMSC 10K Benchmark with Error Bars
Runs 5 independent trials with different random seeds.
Measures both Bit-Perfect (Lossless) and Lossy (float16) modes.
"""
import sys, time
import numpy as np

sys.path.insert(0, ".")
from holodb.codecs.spherical_embedding_codec import SphericalEmbeddingCodec

N_TRIALS = 5
N_VECTORS = 10000
DIM = 1536

results_bp = {
    "ratio": [],
    "encode_s": [],
    "decode_ms": [],
    "compressed_kb": [],
}

results_lossy = {
    "ratio": [],
    "encode_s": [],
    "decode_ms": [],
    "compressed_kb": [],
}

def generate_vectors(n, dim, seed):
    rng = np.random.default_rng(seed)
    # Generate random standard normal, normalize to unit-norm
    vecs_raw = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs_raw, axis=1, keepdims=True)
    vecs = (vecs_raw / norms).tolist()
    return [np.array(v, dtype=np.float32) for v in vecs]

for trial in range(N_TRIALS):
    seed = 42 + trial * 7
    print(f"\n--- Trial {trial+1}/{N_TRIALS} (seed={seed}) ---")
    vecs = generate_vectors(N_VECTORS, DIM, seed)
    orig_kb = N_VECTORS * DIM * 4 / 1024

    # 1. Bit-Perfect (Lossless)
    print("Running PMSC Bit-Perfect...")
    codec_bp = SphericalEmbeddingCodec(
        keep_ratio=0.03, use_alpha=True,
        differential=False, auto_ratio=True, lossy=False,
    )
    t0 = time.time()
    payload_bp = codec_bp.encode(vecs)
    encode_bp_time = time.time() - t0

    # Decode first 10 vectors to average decode time per vector
    t1 = time.time()
    for idx in range(10):
        _ = SphericalEmbeddingCodec.decode(payload_bp, idx)
    decode_bp_time = ((time.time() - t1) / 10) * 1000

    stats_bp = SphericalEmbeddingCodec.compression_stats(vecs, payload_bp)
    results_bp["ratio"].append(stats_bp["ratio"])
    results_bp["encode_s"].append(encode_bp_time)
    results_bp["decode_ms"].append(decode_bp_time)
    results_bp["compressed_kb"].append(stats_bp["compressed_kb"])
    print(f"  BP: ratio={stats_bp['ratio']:.3f}x, encode={encode_bp_time:.2f}s, decode={decode_bp_time:.3f}ms")

    # 2. Lossy (float16)
    print("Running PMSC Lossy (float16)...")
    codec_lossy = SphericalEmbeddingCodec(lossy=True)
    t0 = time.time()
    payload_lossy = codec_lossy.encode(vecs)
    encode_lossy_time = time.time() - t0

    t1 = time.time()
    for idx in range(10):
        _ = SphericalEmbeddingCodec.decode(payload_lossy, idx)
    decode_lossy_time = ((time.time() - t1) / 10) * 1000

    stats_lossy = SphericalEmbeddingCodec.compression_stats(vecs, payload_lossy)
    results_lossy["ratio"].append(stats_lossy["ratio"])
    results_lossy["encode_s"].append(encode_lossy_time)
    results_lossy["decode_ms"].append(decode_lossy_time)
    results_lossy["compressed_kb"].append(stats_lossy["compressed_kb"])
    print(f"  Lossy: ratio={stats_lossy['ratio']:.3f}x, encode={encode_lossy_time:.2f}s, decode={decode_lossy_time:.3f}ms")

print("\n" + "=" * 70)
print("PMSC 10K Benchmark - Error Bars Statistical Summary")
print("=" * 70)

print("\n[PMSC Bit-Perfect (Lossless)]")
for key in results_bp:
    arr = np.array(results_bp[key])
    print(f"  {key:15s}: {arr.mean():.4f} ± {arr.std():.4f}")

print("\n[PMSC Lossy (float16)]")
for key in results_lossy:
    arr = np.array(results_lossy[key])
    print(f"  {key:15s}: {arr.mean():.4f} ± {arr.std():.4f}")

print("=" * 70)
