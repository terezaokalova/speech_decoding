#!/usr/bin/env python3
"""
Generate Supplementary Figures from Real Diphone Logits
========================================================
Creates 3 figures showing model behavior without CTC decoding:
A. Phoneme distribution (predicted counts)
B. Confidence histogram (model certainty)
C. Emission trace (temporal dynamics)

All figures use REAL logits from saved inference outputs.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax

# =============================================================================
# Configuration
# =============================================================================
LOGITS_DIR = "/mnt/sauce/littlab/users/okalova/speech_decoding/results/diphone_run_1/logits"
OUTPUT_DIR = "/mnt/sauce/littlab/users/okalova/braintotext/paper_figures"

# 40 phonemes (indices 1-40 in CTC, index 0 is blank)
PHONEMES = [
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY",
    "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY",
    "P", "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH", "SIL"
]

# Style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['figure.dpi'] = 150

print("=" * 70)
print("GENERATING SUPPLEMENTARY FIGURES FROM REAL LOGITS")
print("=" * 70)
print(f"\nLogits directory: {LOGITS_DIR}")
print(f"Output directory: {OUTPUT_DIR}\n")

# =============================================================================
# Helper: Load Logits
# =============================================================================

def load_all_logits():
    """Load all batch logit files."""
    print("[1/4] Loading diphone logits from saved batches...")

    batch_files = sorted([f for f in os.listdir(LOGITS_DIR) if f.startswith('batch_') and f.endswith('.npy')])
    print(f"  Found {len(batch_files)} batch files")

    all_logits = []
    total_samples = 0

    for batch_file in batch_files:
        logit_path = os.path.join(LOGITS_DIR, batch_file)
        logits = np.load(logit_path)  # Shape: [B, T, 1601]
        all_logits.append(logits)
        total_samples += logits.shape[0]

    print(f"  ✓ Loaded {len(all_logits)} batches")
    print(f"  ✓ Total samples: {total_samples}")
    print(f"  ✓ Total time steps: {sum(l.shape[0] * l.shape[1] for l in all_logits)}")

    return all_logits


def diphone_to_phoneme_probs(diphone_logits):
    """
    Convert diphone logits (1601 classes) to phoneme probabilities (41 classes).

    Uses max-pooling: for each phoneme, take the maximum logit across all
    diphones that END in that phoneme.

    Formula: curr_phoneme_idx = ((diphone_idx - 1) % 40) + 1
    """
    B, T, D = diphone_logits.shape

    # Initialize phoneme logits
    phoneme_logits = np.full((B, T, 41), -np.inf)

    # Copy blank (index 0)
    phoneme_logits[:, :, 0] = diphone_logits[:, :, 0]

    # Max-pool diphones to phonemes
    for d_idx in range(1, 1601):
        curr_p_idx = ((d_idx - 1) % 40) + 1
        phoneme_logits[:, :, curr_p_idx] = np.maximum(
            phoneme_logits[:, :, curr_p_idx],
            diphone_logits[:, :, d_idx]
        )

    # Convert to probabilities
    phoneme_probs = softmax(phoneme_logits, axis=-1)

    return phoneme_probs


# =============================================================================
# FIGURE A: Phoneme Distribution (Predicted Counts)
# =============================================================================

def generate_phoneme_distribution(all_logits):
    """
    Generate bar chart showing predicted phoneme counts.
    Excludes Blank (0) and SIL (40) to focus on speech sounds.
    """
    print("\n[2/4] Generating Figure A: Phoneme Distribution...")

    # Count phoneme predictions across all batches
    phoneme_counts = np.zeros(41)

    for batch_logits in all_logits:
        # Convert to phoneme probabilities
        phoneme_probs = diphone_to_phoneme_probs(batch_logits)

        # Get predictions (argmax)
        preds = np.argmax(phoneme_probs, axis=-1)  # Shape: [B, T]

        # Count each phoneme
        for p_idx in range(41):
            phoneme_counts[p_idx] += (preds == p_idx).sum()

    # Exclude Blank (0) and SIL (40)
    phoneme_counts_speech = phoneme_counts[1:40]  # Indices 1-39
    phoneme_labels = PHONEMES[:39]  # Exclude SIL

    # Sort by count (descending)
    sorted_indices = np.argsort(phoneme_counts_speech)[::-1]
    sorted_counts = phoneme_counts_speech[sorted_indices]
    sorted_labels = [phoneme_labels[i] for i in sorted_indices]

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))

    bars = ax.bar(range(len(sorted_labels)), sorted_counts, color='#3498db', alpha=0.8, edgecolor='black', linewidth=0.8)

    # Highlight top 5
    for i in range(min(5, len(bars))):
        bars[i].set_color('#e74c3c')

    ax.set_xlabel('Phoneme (Sorted by Frequency)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Predicted Count', fontweight='bold', fontsize=12)
    ax.set_title('Figure A: Phoneme Distribution from Model Predictions\n(Excluding Blank and Silence)',
                 fontweight='bold', fontsize=14)
    ax.set_xticks(range(len(sorted_labels)))
    ax.set_xticklabels(sorted_labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # Add count labels on top bars
    for i in range(min(10, len(bars))):
        height = bars[i].get_height()
        ax.text(bars[i].get_x() + bars[i].get_width()/2., height + max(sorted_counts)*0.01,
                f'{int(height)}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()

    outpath = os.path.join(OUTPUT_DIR, 'phoneme_dist.png')
    plt.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  ✓ Saved: {outpath}")
    print(f"  ✓ Total speech phonemes predicted: {int(phoneme_counts_speech.sum())}")
    print(f"  ✓ Blank count: {int(phoneme_counts[0])}")
    print(f"  ✓ SIL count: {int(phoneme_counts[40])}")
    print(f"  ✓ Top 5 phonemes: {', '.join(sorted_labels[:5])}")

    return outpath


# =============================================================================
# FIGURE B: Confidence Histogram
# =============================================================================

def generate_confidence_histogram(all_logits):
    """
    Generate histogram of max probabilities (confidence scores).
    Shows how decisive the model is at each time step.
    """
    print("\n[3/4] Generating Figure B: Confidence Histogram...")

    all_confidences = []

    for batch_logits in all_logits:
        # Convert to phoneme probabilities
        phoneme_probs = diphone_to_phoneme_probs(batch_logits)

        # Get max probability (confidence) at each time step
        max_probs = np.max(phoneme_probs, axis=-1)  # Shape: [B, T]

        # Flatten
        all_confidences.extend(max_probs.flatten().tolist())

    all_confidences = np.array(all_confidences)

    print(f"  ✓ Collected {len(all_confidences)} confidence values")
    print(f"  ✓ Mean confidence: {all_confidences.mean():.3f}")
    print(f"  ✓ Median confidence: {np.median(all_confidences):.3f}")
    print(f"  ✓ High confidence (>0.9): {(all_confidences > 0.9).sum() / len(all_confidences) * 100:.1f}%")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    counts, bins, patches = ax.hist(all_confidences, bins=50, color='#2ecc71', alpha=0.8,
                                      edgecolor='black', linewidth=0.8)

    # Color high confidence bars differently
    for i, patch in enumerate(patches):
        if bins[i] > 0.9:
            patch.set_facecolor('#e74c3c')

    ax.axvline(x=all_confidences.mean(), color='#3498db', linestyle='--', linewidth=2.5,
               label=f'Mean: {all_confidences.mean():.3f}')
    ax.axvline(x=np.median(all_confidences), color='#f39c12', linestyle='--', linewidth=2.5,
               label=f'Median: {np.median(all_confidences):.3f}')

    ax.set_xlabel('Max Probability (Confidence)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Count', fontweight='bold', fontsize=12)
    ax.set_title('Figure B: Model Confidence Distribution\n(Max Probability at Each Time Step)',
                 fontweight='bold', fontsize=14)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    outpath = os.path.join(OUTPUT_DIR, 'confidence_hist.png')
    plt.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  ✓ Saved: {outpath}")

    return outpath


# =============================================================================
# FIGURE C: Emission Trace (Temporal Dynamics)
# =============================================================================

def generate_emission_trace(all_logits):
    """
    Generate heatmap showing top 5 phoneme probabilities over time.
    Uses the first trial from the first batch.
    """
    print("\n[4/4] Generating Figure C: Emission Trace...")

    # Take first trial from first batch
    first_batch_logits = all_logits[0]
    first_trial_logits = first_batch_logits[0:1, :, :]  # Shape: [1, T, 1601]

    # Convert to phoneme probabilities
    phoneme_probs = diphone_to_phoneme_probs(first_trial_logits)  # Shape: [1, T, 41]
    phoneme_probs = phoneme_probs[0, :, :]  # Shape: [T, 41]

    # Take first 100 time steps (2 seconds at 20ms resolution)
    T_max = min(100, phoneme_probs.shape[0])
    phoneme_probs = phoneme_probs[:T_max, :]  # Shape: [T_max, 41]

    # Exclude blank (0)
    phoneme_probs_speech = phoneme_probs[:, 1:]  # Shape: [T_max, 40]

    # Find top 5 most active phonemes (highest average probability)
    avg_probs = phoneme_probs_speech.mean(axis=0)
    top_5_indices = np.argsort(avg_probs)[-5:][::-1]

    # Extract data for top 5 phonemes
    top_5_probs = phoneme_probs_speech[:, top_5_indices].T  # Shape: [5, T_max]
    top_5_labels = [PHONEMES[i] for i in top_5_indices]

    print(f"  ✓ Time steps: {T_max} (duration: {T_max * 0.02:.1f} seconds)")
    print(f"  ✓ Top 5 phonemes: {', '.join(top_5_labels)}")

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))

    im = ax.imshow(top_5_probs, cmap='YlOrRd', aspect='auto', interpolation='nearest', vmin=0, vmax=0.3)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Probability', fontweight='bold', fontsize=11)

    # Axes
    ax.set_xlabel('Time Step (20ms bins)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Phoneme', fontweight='bold', fontsize=12)
    ax.set_title('Figure C: Emission Trace - Top 5 Phonemes Over Time\n(First Trial, First 2 Seconds)',
                 fontweight='bold', fontsize=14)

    # Y-axis labels
    ax.set_yticks(range(len(top_5_labels)))
    ax.set_yticklabels(top_5_labels, fontsize=11)

    # X-axis ticks every 10 steps (0.2 seconds)
    x_ticks = np.arange(0, T_max, 10)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f'{int(t*20)}ms' for t in x_ticks], rotation=45, ha='right')

    plt.tight_layout()

    outpath = os.path.join(OUTPUT_DIR, 'emission_trace.png')
    plt.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  ✓ Saved: {outpath}")

    return outpath


# =============================================================================
# Main
# =============================================================================

def main():
    # Load logits
    all_logits = load_all_logits()

    # Generate figures
    fig_a = generate_phoneme_distribution(all_logits)
    fig_b = generate_confidence_histogram(all_logits)
    fig_c = generate_emission_trace(all_logits)

    # Summary
    print("\n" + "=" * 70)
    print("SUPPLEMENTARY FIGURES GENERATED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"  1. {fig_a}")
    print(f"  2. {fig_b}")
    print(f"  3. {fig_c}")
    print("\n✓ All figures use REAL diphone logits (NO SIMULATION)")
    print("✓ No CTC decoding required - direct probability analysis")
    print("=" * 70)


if __name__ == "__main__":
    main()
