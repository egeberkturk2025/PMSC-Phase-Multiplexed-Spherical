# Phase-Multiplexed Spherical Collapse (PMSC)

This repository contains the core implementation, tests, and academic paper for the **Phase-Multiplexed Spherical Collapse (PMSC)** embedding compression method.

## ⚠️ Copyright and License

**Copyright © 2026 Ege Berk Türk. All rights reserved.**

* This code, theory, and draft paper are the intellectual property of **Ege Berk Türk**.
* **Commercial use of any kind is strictly prohibited.**
* Redistribution and modification for personal, academic, or non-commercial research purposes are permitted under the terms of the `LICENSE` file.

---

## 🚀 Overview

PMSC is a training-free, deterministic embedding compression method that achieves high compression ratios by exploiting the mathematical properties of high-dimensional unit-norm vectors in spherical coordinate systems.

### Key Contributions
1. **Delta-Shift Spherical Transformation**: By transforming vectors to spherical coordinates and subtracting the concentration center ($\pi/2$), we trigger **mantissa collapse** in addition to **exponent collapse**.
2. **FFT Phase Multiplexing**: We reshape the delta angles into 2D matrices and process them in the frequency domain, sharing a common amplitude basis across all vectors in a batch and storing only per-vector phase keys.
3. **Robust Lossy Mode**: A float16-based delta compression path that bypasses FFT quantization, avoiding cumulative trigonometric reconstruction errors and guaranteeing **cosine similarity of 1.0000**.

---

## 📊 Experimental Results

Tested on 50 vectors of dimension $d=1536$:

| Codec | Mode | Ratio | Cosine Similarity | Category |
|:---|:---:|:---:|:---:|:---:|
| **PMSC (Bit-Perfect)** | Lossless (Delta-Perfect) | **3.66×** | 1.0000 | 🥇 Best Ratio |
| **PMSC (Lossy)** | near-Lossless (float16) | **2.09×** | **1.0000** | 🥈 Best Fidelity |
| **EmbeddingCodec** | Baseline | 2.86× | >0.99 | |
| **arXiv 2602.00079** | Academic Baseline | ~1.50× | Lossless | |

### Empirical Validation on Real Semantic Embeddings
We verified the theory using real embeddings generated from 50 diverse sentences via `sentence-transformers`:

#### 1. MiniLM ($d=384$)
* Exponent 127 Ratio: 99.07% (Gaussian) | **98.98% (Real)**
* Bit-Perfect Compression Ratio: 2.63x (Gaussian) | **2.56x (Real)**

#### 2. MPNet ($d=768$)
* Exponent 127 Ratio: 99.50% (Gaussian) | **99.40% (Real)**
* Bit-Perfect Compression Ratio: 3.15x (Gaussian) | **3.06x (Real)**

---

## 📁 Repository Structure

```
PMSC-Phase-Multiplexed-Spherical/
├── holodb/
│   ├── __init__.py
│   └── codecs/
│       ├── __init__.py
│       ├── multiplexed_holographic.py   # FFT phase multiplexing engine
│       └── spherical_embedding_codec.py # Spherical collapse & delta logic
├── tests/
│   ├── __init__.py
│   └── test_spherical_embedding_codec.py# 20+ automated verification tests
├── benchmark.py                         # Benchmark script comparing Gaussian vs Real embs
├── academic_paper_draft.md              # Markdown draft of the publication
├── main.tex                             # LaTeX draft of the publication (arXiv format)
├── main.bib                             # LaTeX bibliography
├── LICENSE                              # License (prohibits commercial use)
└── README.md                            # This file
```

---

## ⚙️ Running the Code

### Dependencies
* Python 3.8+
* NumPy >= 1.24
* Scipy >= 1.10
* SentenceTransformers (for real-world benchmark)
* Pytest (for running verification tests)

Install dependencies:
```bash
pip install numpy scipy sentence-transformers pytest
```

### Running the Verification Tests
To run the full test suite verifying correctness, bit-perfection, and cosine similarity guarantees:
```bash
pytest tests/ -v
```

### Running the Real-World Benchmark
To reproduce the experimental tables comparing Gaussian baseline vs. real semantic embeddings:
```bash
python benchmark.py
```
