import os
import sys
import math
import numpy as np

# Add current directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sentence_transformers import SentenceTransformer
from holodb.codecs.spherical_embedding_codec import SphericalEmbeddingCodec

# 50 diverse real sentences for generating semantic embeddings
real_sentences = [
    "Artificial intelligence is transforming the modern software development landscape.",
    "Quantum computing leverages superposition and entanglement to solve complex problems.",
    "The Mona Lisa is a half-length portrait painting by Italian artist Leonardo da Vinci.",
    "Photosynthesis is the process used by plants to convert light energy into chemical energy.",
    "The theory of general relativity was published by Albert Einstein in 1915.",
    "Deep learning models require large amounts of data and computational power to train.",
    "A database index is a data structure that improves the speed of data retrieval operations.",
    "Climate change is one of the most pressing global challenges of our time.",
    "The Great Wall of China is a series of fortifications built across the historical northern borders.",
    "DNA replication is the biological process of producing two identical replicas of DNA from one.",
    "Microservices architecture allows teams to develop and deploy services independently.",
    "The history of civilization is marked by technological revolutions and cultural changes.",
    "Standard normal distribution has a mean of zero and a standard deviation of one.",
    "Blockchain technology provides a decentralized ledger for secure transactions.",
    "Web browsers interpret HTML, CSS, and JavaScript to render interactive web pages.",
    "The Solar System consists of the Sun and the objects that orbit it, including eight planets.",
    "Photosynthesis occurs in organelles called chloroplasts, which contain chlorophyll.",
    "Restful APIs use HTTP requests to GET, PUT, POST, and DELETE data.",
    "Natural language processing enables computers to understand human language.",
    "The Industrial Revolution began in Great Britain in the late eighteenth century.",
    "Cybersecurity measures are essential to protect sensitive data from unauthorized access.",
    "Newtonian mechanics describes the motion of macroscopic objects under forces.",
    "An ecosystem includes all living things in a given area interacting with each other.",
    "Version control systems like Git allow developers to track changes in code.",
    "Holography is the science and practice of making holograms, which are 3D images.",
    "The Rosetta Stone provided the key to modern understanding of Egyptian hieroglyphs.",
    "Vector databases specialize in storing and querying high-dimensional vector embeddings.",
    "Software testing ensures that application features work as expected under various conditions.",
    "Renewable energy sources include solar, wind, hydro, and geothermal power.",
    "The Mars Rover collects soil samples and takes high-resolution images of the red planet.",
    "A compiler translates source code written in a high-level language into machine code.",
    "Cryptography secures communication in the presence of adversarial third parties.",
    "The internet is a global system of interconnected computer networks.",
    "Human DNA is organized into 23 pairs of chromosomes within the cell nucleus.",
    "The Renaissance was a period of cultural, artistic, and scientific rebirth in Europe.",
    "Machine learning algorithms build models based on sample training data.",
    "A linked list is a linear data structure where elements are stored in nodes.",
    "Symphony No. 9 in D minor is the final complete symphony by Ludwig van Beethoven.",
    "Ocean acidification is the ongoing decrease in the pH of the Earth's oceans.",
    "Docker containers package application code and dependencies into a single image.",
    "The study of economics explores how society manages its scarce resources.",
    "Stochastic gradient descent is an iterative method for optimizing objective functions.",
    "The Amazon rainforest is the world's largest tropical rainforest, famed for biodiversity.",
    "The structure of an atom consists of a nucleus containing protons and neutrons.",
    "Cloud computing provides on-demand availability of computer system resources.",
    "Object-oriented programming is based on the concepts of objects and classes.",
    "The French Revolution was a period of radical social and political upheaval.",
    "An algorithm is a finite sequence of rigorous instructions to solve a problem.",
    "Linear algebra is the branch of mathematics concerning linear equations and matrices.",
    "The human brain contains approximately 86 billion neurons interconnected by synapses."
]

def run_benchmark():
    print("======================================================================")
    print("         REAL EMBEDDING SPHERICAL COLLAPSE BENCHMARK")
    print("======================================================================")
    
    # 1. Generate Uniform Random/Gaussian Unit Vectors (d=384 & d=768)
    rng = np.random.default_rng(42)
    
    # Models to test
    models = ["all-MiniLM-L6-v2"]
    # Try all-mpnet-base-v2 if possible, or just stick to MiniLM
    try:
        print("Loading all-mpnet-base-v2...")
        # Check if we can load it
        model_mpnet = SentenceTransformer("all-mpnet-base-v2")
        models.append("all-mpnet-base-v2")
    except Exception as e:
        print("Could not load all-mpnet-base-v2, running with all-MiniLM-L6-v2 only.")
    
    for model_name in models:
        print(f"\nEvaluating Model: {model_name}")
        model = SentenceTransformer(model_name)
        
        # Real embeddings
        print(f"Generating embeddings for {len(real_sentences)} sentences...")
        real_embs = model.encode(real_sentences)
        # Ensure they are float32 list of vectors
        real_vectors = [vec.astype(np.float32) for vec in real_embs]
        dim = real_vectors[0].shape[0]
        
        # Gaussian unit vectors for baseline comparison (same dimension)
        gaussian_vectors = []
        for _ in range(len(real_sentences)):
            g_vec = rng.standard_normal(dim).astype(np.float32)
            norm = np.linalg.norm(g_vec)
            if norm > 1e-9:
                g_vec /= norm
            gaussian_vectors.append(g_vec)
            
        # 2. Analyze entropy and collapse stats
        real_stats = SphericalEmbeddingCodec.entropy_stats(real_vectors)
        gaussian_stats = SphericalEmbeddingCodec.entropy_stats(gaussian_vectors)
        
        # 3. Print comparison
        print("-" * 70)
        print(f"{'Metric':<30} | {'Gaussian (Baseline)':<20} | {'Real Embeddings':<20}")
        print("-" * 70)
        for key in ["dim", "n_vectors", "angle_mean", "angle_std", "pi_half", "angle_mean_vs_pi2", "delta_max_abs", "delta_std", "delta_entropy_bits", "exponent_127_pct", "theoretical_eps", "collapse_confirmed"]:
            val_g = gaussian_stats.get(key)
            val_r = real_stats.get(key)
            
            # Format outputs
            if isinstance(val_g, float):
                if key == "exponent_127_pct":
                    str_g = f"{val_g:.2f}%"
                    str_r = f"{val_r:.2f}%"
                else:
                    str_g = f"{val_g:.6f}"
                    str_r = f"{val_r:.6f}"
            else:
                str_g = str(val_g)
                str_r = str(val_r)
                
            print(f"{key:<30} | {str_g:<20} | {str_r:<20}")
        print("-" * 70)
        
        # 4. Measure compression ratios
        codec_bp = SphericalEmbeddingCodec(keep_ratio=0.03, lossy=False)
        codec_lossy = SphericalEmbeddingCodec(keep_ratio=0.03, lossy=True)
        
        p_bp_g = codec_bp.encode(gaussian_vectors)
        p_lossy_g = codec_lossy.encode(gaussian_vectors)
        
        p_bp_r = codec_bp.encode(real_vectors)
        p_lossy_r = codec_lossy.encode(real_vectors)
        
        orig_bytes = sum(v.nbytes for v in real_vectors)
        
        ratio_bp_g = orig_bytes / p_bp_g.total_bytes()
        ratio_lossy_g = orig_bytes / p_lossy_g.total_bytes()
        
        ratio_bp_r = orig_bytes / p_bp_r.total_bytes()
        ratio_lossy_r = orig_bytes / p_lossy_r.total_bytes()
        
        print(f"Gaussian PMSC Bit-Perfect Ratio : {ratio_bp_g:.2f}x")
        print(f"Gaussian PMSC Lossy (f16) Ratio : {ratio_lossy_g:.2f}x")
        print(f"Real Embs PMSC Bit-Perfect Ratio: {ratio_bp_r:.2f}x")
        print(f"Real Embs PMSC Lossy (f16) Ratio: {ratio_lossy_r:.2f}x")
        
        # Verify cosine similarity
        cosine_sims = []
        for idx, orig in enumerate(real_vectors):
            recon = SphericalEmbeddingCodec.decode(p_lossy_r, idx)
            sim = SphericalEmbeddingCodec.cosine_similarity(orig, recon)
            cosine_sims.append(sim)
        print(f"Real Embs Lossy Cosine Similarity: min={min(cosine_sims):.6f}, mean={np.mean(cosine_sims):.6f}")

if __name__ == "__main__":
    run_benchmark()
