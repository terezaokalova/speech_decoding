#!/usr/bin/env python3
"""
Aligned Ensemble Evaluation - Fix temporal mismatch between Seed 1 and Seed 2.
Searches for optimal frame shift and applies alignment before averaging.
"""
import torch
import torch.nn.functional as F
import numpy as np
import yaml
import os
import sys
import json
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d
from editdistance import eval as edit_distance

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rnn_model import GRUDecoder
from dataset import BrainToTextDataset
from data_augmentations import gauss_smooth
from torch.utils.data import DataLoader

# CONFIG
SEED1_PATH = "/users/okalova/speech_decoding/results/diphone_run_1/checkpoint/best_checkpoint"
SEED2_PATH = "/users/okalova/speech_decoding/results/diphone_run_2/checkpoint/best_checkpoint"
ARGS_PATH = "/users/okalova/speech_decoding/results/diphone_run_1/checkpoint/args.yaml"
TRIALS_PATH = "/users/okalova/speech_decoding/results/diphone_run_1/train_val_trials.json"

BLANK_IDX = 0
SHIFT_RANGE = [-3, -2, -1, 0, 1, 2, 3]
SMOOTHING_SIGMA = 0.5  # Mild Gaussian smoothing for probability overlap


def load_model(checkpoint_path, args, device):
    """Load a model from checkpoint."""
    model = GRUDecoder(
        neural_dim=args['model']['n_input_features'],
        n_units=args['model']['n_units'],
        n_days=len(args['dataset']['sessions']),
        n_classes=args['dataset']['n_classes'],
        rnn_dropout=args['model']['rnn_dropout'],
        input_dropout=args['model']['input_network']['input_layer_dropout'],
        n_layers=args['model']['n_layers'],
        patch_size=args['model']['patch_size'],
        patch_stride=args['model']['patch_stride'],
    )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt

    clean_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace("module.", "").replace("_orig_mod.", "")
        clean_dict[new_k] = v

    model.load_state_dict(clean_dict)
    model.to(device)
    model.eval()
    return model


def shift_logits(logits, shift, n_classes):
    """
    Shift logits temporally with proper padding (no wrap-around).
    Positive shift = Seed 2 is behind, shift it forward (earlier frames).
    Negative shift = Seed 2 is ahead, shift it backward (later frames).

    Args:
        logits: [B, T, C] tensor
        shift: integer frames to shift
        n_classes: number of classes for blank padding

    Returns:
        shifted logits with blank-padded edges
    """
    if shift == 0:
        return logits

    B, T, C = logits.shape

    # Create blank logits (high confidence on blank token)
    blank_logits = torch.full((B, abs(shift), C), -10.0, device=logits.device, dtype=logits.dtype)
    blank_logits[:, :, BLANK_IDX] = 10.0  # High confidence blank

    if shift > 0:
        # Shift forward: remove first `shift` frames, pad at end
        shifted = torch.cat([logits[:, shift:, :], blank_logits], dim=1)
    else:
        # Shift backward: pad at beginning, remove last `shift` frames
        shifted = torch.cat([blank_logits, logits[:, :shift, :]], dim=1)

    return shifted


def apply_temporal_smoothing(probs, sigma):
    """Apply Gaussian smoothing along time axis to probabilities."""
    # probs: [B, T, C] numpy array
    if sigma <= 0:
        return probs

    smoothed = np.zeros_like(probs)
    for b in range(probs.shape[0]):
        for c in range(probs.shape[2]):
            smoothed[b, :, c] = gaussian_filter1d(probs[b, :, c], sigma=sigma)

    # Renormalize to ensure valid probability distribution
    smoothed = smoothed / (smoothed.sum(axis=-1, keepdims=True) + 1e-10)
    return smoothed


def main():
    print("=" * 70)
    print("ALIGNED ENSEMBLE EVALUATION")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Load args and models
    with open(ARGS_PATH, 'r') as f:
        args = yaml.safe_load(f)

    print("Loading models...")
    model1 = load_model(SEED1_PATH, args, device)
    model2 = load_model(SEED2_PATH, args, device)
    print("  Both models loaded successfully\n")

    # Load validation dataset
    with open(TRIALS_PATH, 'r') as f:
        trials = json.load(f)
    val_trials = {int(k): v for k, v in trials['val'].items()}

    dataset = BrainToTextDataset(
        trial_indicies=val_trials,
        split='test',
        days_per_batch=None,
        n_batches=None,
        batch_size=args['dataset']['batch_size'],
        must_include_days=None,
        feature_subset=None
    )

    val_loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)
    print(f"Loaded {len(dataset)} validation batches\n")

    # Get transform parameters
    transform_args = args['dataset']['data_transforms']
    smooth_data = transform_args.get('smooth_data', True)
    smooth_kernel_std = transform_args.get('smooth_kernel_std', 2)
    smooth_kernel_size = transform_args.get('smooth_kernel_size', 100)
    n_classes = args['dataset']['n_classes']
    patch_size = args['model']['patch_size']
    patch_stride = args['model']['patch_stride']

    # =========================================================================
    # STEP 1: ALIGNMENT SEARCH (first 10 batches)
    # =========================================================================
    print("STEP 1: ALIGNMENT SEARCH")
    print("-" * 70)
    print(f"Testing shifts: {SHIFT_RANGE}")

    # Collect logits from first 10 batches for alignment search
    alignment_batches = []
    batch_count = 0

    with torch.no_grad():
        for batch in val_loader:
            day = batch['day_indicies'][0].item()
            if args['dataset']['dataset_probability_val'][day] == 0:
                continue

            features = batch['input_features'].to(device)

            if smooth_data:
                features = gauss_smooth(
                    inputs=features,
                    device=device,
                    smooth_kernel_std=smooth_kernel_std,
                    smooth_kernel_size=smooth_kernel_size,
                )

            with torch.autocast(device_type="cuda", enabled=args['use_amp'], dtype=torch.bfloat16):
                log1 = model1(features, batch['day_indicies'].to(device))
                log2 = model2(features, batch['day_indicies'].to(device))

            alignment_batches.append((log1.float().cpu(), log2.float().cpu()))

            batch_count += 1
            if batch_count >= 10:
                break

    print(f"Alignment search using {batch_count} batches\n")

    # Test each shift by computing correlation across all batches
    shift_correlations = {}
    for shift in SHIFT_RANGE:
        all_corrs = []
        for log1, log2 in alignment_batches:
            shifted_log2 = shift_logits(log2, shift, n_classes)

            # Flatten and compute correlation for this batch
            flat1 = log1.numpy().flatten()
            flat2 = shifted_log2.numpy().flatten()

            corr, _ = pearsonr(flat1, flat2)
            all_corrs.append(corr)

        avg_corr = np.mean(all_corrs)
        shift_correlations[shift] = avg_corr
        print(f"  Shift {shift:+d}: avg correlation = {avg_corr:.6f}")

    # Find best shift
    best_shift = max(shift_correlations, key=shift_correlations.get)
    best_corr = shift_correlations[best_shift]

    print(f"\n{'='*70}")
    print(f"OPTIMAL TEMPORAL SHIFT DETECTED: {best_shift:+d} frames")
    print(f"Correlation improvement: {shift_correlations[0]:.4f} -> {best_corr:.4f}")
    print(f"{'='*70}\n")

    # =========================================================================
    # STEP 2: FULL EVALUATION WITH ALIGNMENT
    # =========================================================================
    print("STEP 2: FULL EVALUATION")
    print("-" * 70)
    print(f"Applying shift={best_shift:+d} to Seed 2")
    print(f"Smoothing sigma={SMOOTHING_SIGMA}\n")

    # Reset loader
    val_loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)

    # Metrics accumulators
    total_ed_m1 = 0
    total_ed_m2 = 0
    total_ed_unaligned = 0
    total_ed_aligned = 0
    total_ed_aligned_smooth = 0
    total_seq_len = 0
    n_trials = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            day = batch['day_indicies'][0].item()
            if args['dataset']['dataset_probability_val'][day] == 0:
                continue

            features = batch['input_features'].to(device)
            labels = batch['seq_class_ids'].to(device)
            n_time_steps = batch['n_time_steps'].to(device)
            phone_seq_lens = batch['phone_seq_lens'].to(device)
            day_indicies = batch['day_indicies'].to(device)

            if smooth_data:
                features = gauss_smooth(
                    inputs=features,
                    device=device,
                    smooth_kernel_std=smooth_kernel_std,
                    smooth_kernel_size=smooth_kernel_size,
                )

            adjusted_lens = ((n_time_steps - patch_size) / patch_stride + 1).to(torch.int32)

            with torch.autocast(device_type="cuda", enabled=args['use_amp'], dtype=torch.bfloat16):
                logits1 = model1(features, day_indicies)
                logits2 = model2(features, day_indicies)

            logits1 = logits1.float()
            logits2 = logits2.float()

            # Apply alignment shift to Seed 2
            logits2_aligned = shift_logits(logits2, best_shift, n_classes)

            # Compute probabilities
            probs1 = F.softmax(logits1, dim=-1)
            probs2 = F.softmax(logits2, dim=-1)
            probs2_aligned = F.softmax(logits2_aligned, dim=-1)

            # Unaligned ensemble (baseline)
            probs_unaligned = (probs1 + probs2) / 2.0

            # Aligned ensemble (no smoothing)
            probs_aligned = (probs1 + probs2_aligned) / 2.0

            # Aligned ensemble with smoothing
            probs1_np = probs1.cpu().numpy()
            probs2_aligned_np = probs2_aligned.cpu().numpy()

            probs1_smooth = apply_temporal_smoothing(probs1_np, SMOOTHING_SIGMA)
            probs2_smooth = apply_temporal_smoothing(probs2_aligned_np, SMOOTHING_SIGMA)
            probs_aligned_smooth = (probs1_smooth + probs2_smooth) / 2.0
            probs_aligned_smooth = torch.tensor(probs_aligned_smooth, device=device)

            # Decode and evaluate each trial
            for iter_idx in range(features.shape[0]):
                adj_len = adjusted_lens[iter_idx].item()
                true_seq = labels[iter_idx][:phone_seq_lens[iter_idx]].cpu().numpy()

                def decode(probs_tensor):
                    decoded = torch.argmax(probs_tensor[iter_idx, :adj_len, :], dim=-1)
                    decoded = torch.unique_consecutive(decoded)
                    decoded = decoded.cpu().numpy()
                    decoded = np.array([i for i in decoded if i != 0])
                    return decoded

                dec_m1 = decode(probs1)
                dec_m2 = decode(probs2)
                dec_unaligned = decode(probs_unaligned)
                dec_aligned = decode(probs_aligned)
                dec_aligned_smooth = decode(probs_aligned_smooth)

                total_ed_m1 += edit_distance(dec_m1, true_seq)
                total_ed_m2 += edit_distance(dec_m2, true_seq)
                total_ed_unaligned += edit_distance(dec_unaligned, true_seq)
                total_ed_aligned += edit_distance(dec_aligned, true_seq)
                total_ed_aligned_smooth += edit_distance(dec_aligned_smooth, true_seq)
                total_seq_len += len(true_seq)
                n_trials += 1

            if (batch_idx + 1) % 20 == 0:
                per_m1 = total_ed_m1 / total_seq_len
                per_aligned = total_ed_aligned / total_seq_len
                print(f"  Batch {batch_idx + 1}: M1={per_m1:.4f}, Aligned={per_aligned:.4f}")

    # =========================================================================
    # FINAL RESULTS
    # =========================================================================
    per_m1 = total_ed_m1 / total_seq_len
    per_m2 = total_ed_m2 / total_seq_len
    per_unaligned = total_ed_unaligned / total_seq_len
    per_aligned = total_ed_aligned / total_seq_len
    per_aligned_smooth = total_ed_aligned_smooth / total_seq_len

    best_single = min(per_m1, per_m2)

    def improvement(per):
        return (best_single - per) / best_single * 100

    print("\n" + "=" * 70)
    print("ALIGNED ENSEMBLE RESULTS")
    print("=" * 70)
    print(f"Trials evaluated: {n_trials}")
    print(f"Total phonemes:   {total_seq_len}")
    print(f"Optimal shift:    {best_shift:+d} frames")
    print("-" * 70)
    print(f"{'Method':<35} {'PER':<12} {'Improvement':<15}")
    print("-" * 70)
    print(f"{'Seed 1 (baseline)':<35} {per_m1:.4f} ({per_m1*100:.2f}%)  --")
    print(f"{'Seed 2':<35} {per_m2:.4f} ({per_m2*100:.2f}%)  --")
    print(f"{'Unaligned Ensemble (broken)':<35} {per_unaligned:.4f} ({per_unaligned*100:.2f}%)  {improvement(per_unaligned):+.2f}%")
    print(f"{'Aligned Ensemble (shift={best_shift:+d})':<35} {per_aligned:.4f} ({per_aligned*100:.2f}%)  {improvement(per_aligned):+.2f}%")
    print(f"{'Aligned + Smoothed (σ={SMOOTHING_SIGMA})':<35} {per_aligned_smooth:.4f} ({per_aligned_smooth*100:.2f}%)  {improvement(per_aligned_smooth):+.2f}%")
    print("=" * 70)

    # Verdict
    print("\nVERDICT:")
    if per_aligned < best_single:
        print(f"✓ ALIGNMENT SUCCESSFUL! PER improved from {best_single:.4f} to {per_aligned:.4f}")
        print(f"  Relative improvement: {improvement(per_aligned):.2f}%")
    elif per_aligned_smooth < best_single:
        print(f"✓ ALIGNMENT + SMOOTHING SUCCESSFUL! PER improved from {best_single:.4f} to {per_aligned_smooth:.4f}")
        print(f"  Relative improvement: {improvement(per_aligned_smooth):.2f}%")
    else:
        print(f"✗ Alignment did not improve PER. Best single model ({best_single:.4f}) is still optimal.")
        print(f"  The temporal mismatch may not be the primary issue.")
    print("=" * 70)

    return per_aligned


if __name__ == "__main__":
    main()
