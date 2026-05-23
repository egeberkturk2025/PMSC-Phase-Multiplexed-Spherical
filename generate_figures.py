import matplotlib.pyplot as plt
import numpy as np

# Data from our official dimensions scaling benchmark
dimensions = [256, 384, 512, 768, 1024, 1536, 3072]
pmsc_bp = [2.42, 2.55, 2.77, 3.03, 3.21, 3.57, 4.07]
pmsc_lossy = [1.92, 1.95, 1.98, 2.03, 2.07, 2.09, 2.13]
baseline_xiao = [1.50] * len(dimensions)

# Create plotting environment
plt.figure(figsize=(9, 5.5))
plt.style.use('seaborn-v0_8-whitegrid')

# Plot lines
plt.plot(dimensions, pmsc_bp, marker='o', linewidth=2.5, color='#1f77b4', label='PMSC (Bit-Perfect Lossless)')
plt.plot(dimensions, pmsc_lossy, marker='s', linewidth=2.0, color='#ff7f0e', label='PMSC Lossy (float16)')
plt.plot(dimensions, baseline_xiao, linestyle='--', linewidth=1.5, color='#d62728', label='Academic Baseline (Xiao 2026)')

# Title & labels
plt.title('PMSC Compression Ratio vs. Embedding Dimension', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Vector Dimension (d)', fontsize=12)
plt.ylabel('Compression Ratio (x)', fontsize=12)

# Set ticks and logarithmic-like scaling representation
plt.xticks(dimensions, [str(d) for d in dimensions])
plt.grid(True, which='both', linestyle=':', alpha=0.6)

# Annotate key values
for x, y in zip(dimensions, pmsc_bp):
    plt.annotate(f"{y:.2f}x", (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, fontweight='bold', color='#1f77b4')

for x, y in zip(dimensions, pmsc_lossy):
    plt.annotate(f"{y:.2f}x", (x, y), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=9, color='#ff7f0e')

# Adjust layout & Legend
plt.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9, shadow=True, fontsize=11)
plt.tight_layout()

# Save the plot
output_path = 'compression_scaling.png'
plt.savefig(output_path, dpi=300)
print(f"Figure successfully saved as {output_path}")
