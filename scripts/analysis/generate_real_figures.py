#!/usr/bin/env python3
"""
Generate REAL Figures from Actual Model Outputs
================================================
Uses actual logits, labels, and training logs - NO SIMULATED DATA.

Data Sources:
- Logits: results/diphone_run_1/logits/batch_*.npy
- Labels: Validation dataset (re-instantiated)
- Training: results/diphone_run_1/training_log
"""

import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from pathlib import Path

# Add paths for importing dataset
sys.path.insert(0, '/mnt/sauce/littlab/users/okalova/speech_decoding/code/model_training_diphone')
from dataset import BrainToTextDataset, train_test_split_indicies

# =============================================================================
# Configuration
# =============================================================================
RESULTS_DIR = "/mnt/sauce/littlab/users/okalova/speech_decoding/results"
LOGITS_DIR = f"{RESULTS_DIR}/diphone_run_1/logits"
TRAINING_LOG = f"{RESULTS_DIR}/diphone_run_1/training_log"
DATASET_DIR = "/mnt/sauce/littlab/users/okalova/speech_decoding/data"
OUTPUT_DIR = "/mnt/sauce/littlab/users/okalova/braintotext/paper_figures"

# Phoneme vocabulary (40 phonemes)
PHONEMES = [
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY",
    "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY",
    "P", "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH", "SIL"
]

# Style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.dpi'] = 150

# =============================================================================
# CONFIRMED: Script uses DUMMY DATA!
# =============================================================================
print("=" * 70)
print("⚠️  ANALYSIS: generate_phonetics_figures.py uses SIMULATED DATA!")
print("=" * 70)
print("\nDummy data found at:")
print("  - Line 200: generate_simulated_confusion_matrix() with np.random.uniform")
print("  - Line 474: Simulated PCA with np.random.randn")
print("  - Line 386: Simulated trajectories with np.random")
print("\n✓ NO val_metrics.pkl loading detected")
print("✓ NO real model predictions used")
print("\n" + "=" * 70)
print("GENERATING REAL FIGURES FROM ACTUAL MODEL OUTPUTS")
print("=" * 70 + "\n")

# =============================================================================
# Function: Load Logits and Labels
# =============================================================================

def load_validation_dataset():
    """Re-instantiate the validation dataset to get true labels."""
    print("[1/4] Loading validation dataset...")

    # Session list from args.yaml
    sessions = [
        't15.2023.08.11', 't15.2023.08.13', 't15.2023.08.18', 't15.2023.08.20',
        't15.2023.08.25', 't15.2023.08.27', 't15.2023.09.01', 't15.2023.09.03',
        't15.2023.09.24', 't15.2023.09.29', 't15.2023.10.01', 't15.2023.10.06',
        't15.2023.10.08', 't15.2023.10.13', 't15.2023.10.15', 't15.2023.10.20',
        't15.2023.10.22', 't15.2023.11.03', 't15.2023.11.04', 't15.2023.11.17',
        't15.2023.11.19', 't15.2023.11.26', 't15.2023.12.03', 't15.2023.12.08',
        't15.2023.12.10', 't15.2023.12.17', 't15.2023.12.29', 't15.2024.02.25',
        't15.2024.03.03', 't15.2024.03.08', 't15.2024.03.15', 't15.2024.03.17',
        't15.2024.04.25', 't15.2024.04.28', 't15.2024.05.10', 't15.2024.06.14',
        't15.2024.07.19', 't15.2024.07.21', 't15.2024.07.28', 't15.2025.01.10',
        't15.2025.01.12', 't15.2025.03.14', 't15.2025.03.16', 't15.2025.03.30',
        't15.2025.04.13'
    ]

    # Build file paths
    file_paths = [os.path.join(DATASET_DIR, f'{session}/data_train.hdf5') for session in sessions]

    # Create train/test split
    train_trials, test_trials = train_test_split_indicies(
        file_paths,
        test_percentage=0.1,
        seed=1
    )

    # Create validation dataset (test split)
    val_dataset = BrainToTextDataset(
        trial_indicies=test_trials,
        n_batches=1000,  # Arbitrary, we'll load all data
        split='test',
        batch_size=64,
        days_per_batch=1,
        random_seed=1
    )

    print(f"  ✓ Loaded {val_dataset.n_trials} validation trials from {val_dataset.n_days} days")

    return val_dataset


def load_logits_and_labels(val_dataset):
    """Load all logit files and corresponding labels."""
    print("\n[2/4] Loading logits and extracting labels...")

    # Find all batch files
    batch_files = sorted([f for f in os.listdir(LOGITS_DIR) if f.startswith('batch_') and f.endswith('.npy')])
    print(f"  Found {len(batch_files)} logit batch files")

    all_logits = []
    all_labels = []
    all_preds = []

    # Load validation batches to get labels
    print(f"  Loading labels from validation dataset...")

    for batch_idx in range(len(batch_files)):
        # Load logits
        logit_path = os.path.join(LOGITS_DIR, f'batch_{batch_idx:04d}.npy')
        logits = np.load(logit_path)  # Shape: [B, T, 1601]

        # Load corresponding validation batch
        try:
            val_batch = val_dataset[batch_idx]
            labels = val_batch['seq_class_ids'].numpy()  # Shape: [B, T]

            # Get predictions (argmax over diphone dimension)
            preds = np.argmax(logits, axis=-1)  # Shape: [B, T]

            all_logits.append(logits)
            all_labels.append(labels)
            all_preds.append(preds)

        except IndexError:
            print(f"  ⚠️  Batch {batch_idx} out of range, stopping")
            break

    print(f"  ✓ Loaded {len(all_logits)} batches")
    print(f"  ✓ Total samples: {sum(l.shape[0] for l in all_logits)}")

    return all_logits, all_labels, all_preds


def collapse_diphones_to_phonemes(diphone_logits):
    """
    Convert diphone logits (1601 classes) to phoneme logits (41 classes).

    Formula: curr_phoneme_idx = ((diphone_idx - 1) % 40) + 1
    """
    B, T, D = diphone_logits.shape
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

    return phoneme_logits


# =============================================================================
# Figure 1: REAL Confusion Matrix
# =============================================================================

def generate_real_confusion_matrix(all_logits, all_labels):
    """Generate confusion matrix from actual model predictions vs true labels."""
    print("\n[3/4] Generating REAL confusion matrix...")

    # Collapse diphones to phonemes
    print("  Converting diphone logits to phoneme predictions...")

    all_true = []
    all_pred = []

    for batch_logits, batch_labels in zip(all_logits, all_labels):
        # Collapse to phonemes
        phoneme_logits = collapse_diphones_to_phonemes(batch_logits)
        phoneme_preds = np.argmax(phoneme_logits, axis=-1)  # Shape: [B, T]

        # Check shapes match
        B_logits, T_logits = phoneme_preds.shape
        B_labels, T_labels = batch_labels.shape

        # Use minimum batch size to avoid index errors
        B = min(B_logits, B_labels)

        # Flatten and filter out padding (label=0) and blank predictions
        for i in range(B):
            seq_len = min((batch_labels[i] != 0).sum(), T_logits)

            true_seq = batch_labels[i, :seq_len]
            pred_seq = phoneme_preds[i, :seq_len]

            # Filter out blank predictions (0)
            mask = pred_seq != 0

            all_true.extend(true_seq[mask].tolist())
            all_pred.extend(pred_seq[mask].tolist())

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    print(f"  ✓ Collected {len(all_true)} phoneme predictions (non-blank)")

    # Build confusion matrix (41 classes: 0=blank, 1-40=phonemes)
    confusion = np.zeros((41, 41))

    for true_idx, pred_idx in zip(all_true, all_pred):
        if 0 <= true_idx < 41 and 0 <= pred_idx < 41:
            confusion[true_idx, pred_idx] += 1

    # Normalize by row (ground truth)
    row_sums = confusion.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    confusion_norm = confusion / row_sums

    # Focus on actual phonemes (indices 1-40, skip blank at 0)
    confusion_phonemes = confusion_norm[1:41, 1:41]

    # Select top 15 most frequent phonemes for visualization
    phoneme_counts = confusion[1:41, :].sum(axis=1)
    top_indices = np.argsort(phoneme_counts)[-15:][::-1]

    confusion_subset = confusion_phonemes[np.ix_(top_indices, top_indices)]
    phoneme_labels = [PHONEMES[i] for i in top_indices]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))

    sns.heatmap(
        confusion_subset,
        annot=True,
        fmt='.2f',
        cmap='YlOrRd',
        vmin=0,
        vmax=1.0,
        xticklabels=phoneme_labels,
        yticklabels=phoneme_labels,
        square=True,
        cbar_kws={'label': 'Confusion Probability'},
        ax=ax
    )

    ax.set_xlabel('Predicted Phoneme', fontweight='bold', fontsize=12)
    ax.set_ylabel('True Phoneme (Ground Truth)', fontweight='bold', fontsize=12)
    ax.set_title('REAL Confusion Matrix: Diphone Model Predictions\n(Top 15 Most Frequent Phonemes)',
                 fontweight='bold', fontsize=14)

    plt.tight_layout()

    outpath = os.path.join(OUTPUT_DIR, 'real_confusion.png')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  ✓ Saved: {outpath}")

    # Compute overall accuracy
    accuracy = np.trace(confusion_subset) / confusion_subset.sum()
    print(f"  ✓ Phoneme accuracy (top 15): {accuracy*100:.2f}%")

    return outpath


# =============================================================================
# Figure 2: REAL Training Curves
# =============================================================================

def parse_training_log():
    """Parse training log for real loss and PER curves."""
    print("\n[4/4] Parsing REAL training curves from log...")

    batches = []
    val_losses = []
    val_pers = []

    with open(TRAINING_LOG, 'r') as f:
        for line in f:
            # Match: "Val batch 2000: PER (avg): 0.1234 CTC Loss (avg): 12.345"
            match = re.search(r'Val batch (\d+): PER \(avg\): ([\d.]+) CTC Loss \(avg\): ([\d.]+)', line)
            if match:
                batch = int(match.group(1))
                per = float(match.group(2))
                loss = float(match.group(3))

                batches.append(batch)
                val_pers.append(per)
                val_losses.append(loss)

    print(f"  ✓ Extracted {len(batches)} validation checkpoints")

    return np.array(batches), np.array(val_losses), np.array(val_pers)


def generate_real_training_curves():
    """Generate training curves from actual training log."""

    batches, val_losses, val_pers = parse_training_log()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Top: CTC Loss
    ax1.plot(batches, val_losses, linewidth=2.5, color='#e74c3c', marker='o', markersize=4, alpha=0.8)
    ax1.set_ylabel('CTC Loss (Validation)', fontweight='bold', fontsize=12)
    ax1.set_title('REAL Training Curves: Diphone Model', fontweight='bold', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, batches.max() * 1.05)

    # Bottom: PER
    ax2.plot(batches, val_pers * 100, linewidth=2.5, color='#3498db', marker='s', markersize=4, alpha=0.8)
    ax2.set_xlabel('Training Batch', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Phoneme Error Rate (%) - Validation', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, batches.max() * 1.05)

    # Highlight best PER
    best_per_idx = np.argmin(val_pers)
    best_per = val_pers[best_per_idx] * 100
    best_batch = batches[best_per_idx]

    ax2.axhline(y=best_per, color='#27ae60', linestyle='--', alpha=0.5, linewidth=2)
    ax2.text(batches[-1] * 0.6, best_per + 1,
             f'Best: {best_per:.2f}% @ batch {best_batch}',
             fontsize=11, color='#27ae60', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    outpath = os.path.join(OUTPUT_DIR, 'real_training_curves.png')
    plt.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  ✓ Saved: {outpath}")
    print(f"  ✓ Best validation PER: {best_per:.2f}% @ batch {best_batch}")

    return outpath


# =============================================================================
# Main
# =============================================================================

def main():
    # Load data
    val_dataset = load_validation_dataset()
    all_logits, all_labels, all_preds = load_logits_and_labels(val_dataset)

    # Generate figures
    confusion_path = generate_real_confusion_matrix(all_logits, all_labels)
    training_path = generate_real_training_curves()

    # PCA check
    print("\n" + "-" * 70)
    print("⚠️  PCA ANALYSIS:")
    print("-" * 70)
    print("Hidden states were NOT saved in batch_*.npy files (only logits)")
    print("To generate PCA figures, you need to:")
    print("  1. Modify run_inference.py to save hidden states")
    print("  2. Re-run inference with save_hidden_states=True")
    print("  3. Load hidden states here and run PCA")
    print("\nSKIPPING PCA figure for now.")

    # Summary
    print("\n" + "=" * 70)
    print("REAL FIGURES GENERATED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"  1. {confusion_path}")
    print(f"  2. {training_path}")
    print("\n✓ All figures use ACTUAL model outputs (NO SIMULATION)")
    print("=" * 70)


if __name__ == "__main__":
    main()
