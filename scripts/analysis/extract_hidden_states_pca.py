#!/usr/bin/env python3
"""
Extract hidden states from trained diphone model and visualize with PCA by manner of articulation.

This script:
1. Loads the trained diphone model from checkpoint
2. Runs inference on validation data, extracting hidden states from the final GRU layer
3. Maps predicted diphones to phonemes and then to manner of articulation classes
4. Performs PCA on the hidden states
5. Creates a publication-quality visualization colored by manner class
"""

import sys
import os

# Add the model training directory to path
sys.path.insert(0, '/users/okalova/speech_decoding/code/model_training_diphone')

import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict

# Import model and data utilities
from rnn_model import GRUDecoder
from dataset import BrainToTextDataset, train_test_split_indicies
from phoneme_vocab import PHONEME_LIST
from phoneme_to_features import PHONEME_MAP, MANNER_CLASSES

# Configuration
CHECKPOINT_PATH = '/users/okalova/speech_decoding/results/diphone_run_2/checkpoint/best_checkpoint'
OUTPUT_PATH = '/users/okalova/speech_decoding/paper_figures/pca_manner_real.png'
MAX_FRAMES = 10000  # Sample up to this many frames total
NUM_BASE_UNITS = 40  # 39 phonemes + 1 silence

# Model parameters from args.yaml
MODEL_CONFIG = {
    'neural_dim': 512,
    'n_units': 768,
    'n_days': 45,
    'n_classes': 1601,
    'rnn_dropout': 0.4,
    'input_dropout': 0.2,
    'n_layers': 5,
    'patch_size': 14,
    'patch_stride': 4,
}

# Sessions list (from args.yaml)
SESSIONS = [
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

# Which days to validate on (indices where value is 1)
VAL_DAYS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 24, 25, 26, 29, 30, 31, 34, 35, 36, 37, 38, 39, 40,
            41, 42, 43, 44]

DATASET_DIR = '/users/okalova/speech_decoding/data'

# Manner class groupings for cleaner visualization
MANNER_GROUPS = {
    'Stop': ['stop'],
    'Fricative': ['fricative', 'affricate'],
    'Nasal': ['nasal'],
    'Approximant': ['liquid', 'glide'],
    'Vowel': ['vowel_high', 'vowel_mid', 'vowel_low', 'diphthong'],
}

# Map each manner class to its group
MANNER_TO_GROUP = {}
for group, classes in MANNER_GROUPS.items():
    for cls in classes:
        MANNER_TO_GROUP[cls] = group

# Colors for each group
GROUP_COLORS = {
    'Stop': '#E74C3C',      # Red
    'Fricative': '#F39C12', # Orange
    'Nasal': '#3498DB',     # Blue
    'Approximant': '#9B59B6', # Purple
    'Vowel': '#2ECC71',     # Green
}


def diphone_to_phoneme(diphone_idx):
    """Convert diphone CTC index to current phoneme base index.

    Diphone index layout:
    - 0 = CTC blank
    - 1-1600 = diphones where idx-1 = (prev*40) + curr

    Returns:
        Base index (0-39) of the current phoneme, or -1 for blank
    """
    if diphone_idx == 0:
        return -1  # CTC blank
    base_idx = diphone_idx - 1
    curr_phoneme = base_idx % NUM_BASE_UNITS
    return curr_phoneme


def get_manner_group(phoneme_str):
    """Get manner group for a phoneme string."""
    phoneme_upper = phoneme_str.upper()
    if phoneme_upper in PHONEME_MAP:
        manner = PHONEME_MAP[phoneme_upper]['manner']
        if manner == 'silence':
            return None  # Skip silence
        return MANNER_TO_GROUP.get(manner, None)
    return None


def load_model(checkpoint_path, device):
    """Load the trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")

    # Initialize model
    model = GRUDecoder(
        neural_dim=MODEL_CONFIG['neural_dim'],
        n_units=MODEL_CONFIG['n_units'],
        n_days=MODEL_CONFIG['n_days'],
        n_classes=MODEL_CONFIG['n_classes'],
        rnn_dropout=MODEL_CONFIG['rnn_dropout'],
        input_dropout=MODEL_CONFIG['input_dropout'],
        n_layers=MODEL_CONFIG['n_layers'],
        patch_size=MODEL_CONFIG['patch_size'],
        patch_stride=MODEL_CONFIG['patch_stride'],
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle compiled model state dict (keys have '_orig_mod.' prefix)
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        # Remove '_orig_mod.' prefix if present
        new_key = k.replace('_orig_mod.', '')
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    print(f"Model loaded successfully. Val PER: {checkpoint.get('val_PER', 'N/A')}")
    return model


def setup_data_loader(batch_size=32):
    """Set up the validation data loader."""
    print("Setting up data loader...")

    val_file_paths = [
        os.path.join(DATASET_DIR, s, 'data_val.hdf5') for s in SESSIONS
    ]

    # Get validation trial indices
    _, val_trials = train_test_split_indicies(
        file_paths=val_file_paths,
        test_percentage=1,
        seed=1,
        bad_trials_dict=None,
    )

    # Create validation dataset
    val_dataset = BrainToTextDataset(
        trial_indicies=val_trials,
        split='test',
        days_per_batch=None,
        n_batches=None,
        batch_size=batch_size,
        must_include_days=None,
        random_seed=1,
        feature_subset=None,
    )

    print(f"Validation dataset has {len(val_dataset)} batches")
    return val_dataset


def extract_hidden_states(model, dataset, device, max_frames=10000):
    """Extract hidden states and their corresponding manner labels.

    Returns:
        hidden_states: np.array of shape (N, 768)
        manner_labels: list of manner group names
    """
    print("Extracting hidden states...")

    hidden_states_list = []
    manner_labels = []
    frames_collected = 0

    # Storage for captured hidden states from hook
    captured_states = []

    def hook_fn(module, input, output):
        """Hook to capture GRU output."""
        # output is (output_sequence, h_n) for GRU
        # output_sequence has shape (batch, seq_len, hidden_dim)
        captured_states.append(output[0].detach().cpu())

    # Register hook on GRU
    hook = model.gru.register_forward_hook(hook_fn)

    try:
        for batch_idx in range(len(dataset)):
            if frames_collected >= max_frames:
                break

            batch = dataset[batch_idx]

            # Check if this day should be validated
            day_idx = batch['day_indicies'][0].item()
            if day_idx not in VAL_DAYS:
                continue

            # Move data to device
            features = batch['input_features'].to(device)
            n_time_steps = batch['n_time_steps'].to(device)
            day_indices = batch['day_indicies'].to(device)

            # Clear captured states
            captured_states.clear()

            with torch.no_grad():
                # Forward pass (hook captures hidden states)
                logits = model(features, day_indices)

            # Get the captured hidden states
            if not captured_states:
                continue

            hidden = captured_states[0].numpy()  # (batch, seq_len, 768)
            logits_np = logits.cpu().numpy()  # (batch, seq_len, 1601)

            # Calculate adjusted sequence lengths
            patch_size = MODEL_CONFIG['patch_size']
            patch_stride = MODEL_CONFIG['patch_stride']
            adjusted_lens = ((n_time_steps - patch_size) / patch_stride + 1).to(torch.int32).cpu().numpy()

            # Process each sample in batch
            for i in range(hidden.shape[0]):
                seq_len = adjusted_lens[i]

                for t in range(seq_len):
                    if frames_collected >= max_frames:
                        break

                    # Get argmax prediction (diphone index)
                    diphone_idx = np.argmax(logits_np[i, t, :])

                    # Skip blank predictions
                    if diphone_idx == 0:
                        continue

                    # Convert diphone to phoneme
                    phoneme_idx = diphone_to_phoneme(diphone_idx)
                    if phoneme_idx < 0:
                        continue

                    # Get phoneme string
                    phoneme_str = PHONEME_LIST[phoneme_idx]

                    # Get manner group
                    manner_group = get_manner_group(phoneme_str)
                    if manner_group is None:
                        continue  # Skip silence

                    # Store hidden state and label
                    hidden_states_list.append(hidden[i, t, :])
                    manner_labels.append(manner_group)
                    frames_collected += 1

            if batch_idx % 50 == 0:
                print(f"  Processed {batch_idx}/{len(dataset)} batches, {frames_collected} frames collected")

    finally:
        hook.remove()

    print(f"Collected {len(hidden_states_list)} frames total")

    # Stack into array
    hidden_states = np.stack(hidden_states_list, axis=0)

    return hidden_states, manner_labels


def run_pca_and_plot(hidden_states, manner_labels, output_path):
    """Run PCA on hidden states and create visualization."""
    print("Running PCA...")

    # Fit PCA
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(hidden_states)

    explained_var = pca.explained_variance_ratio_
    print(f"Explained variance: PC1={explained_var[0]:.3f}, PC2={explained_var[1]:.3f}")
    print(f"Total explained: {sum(explained_var):.3f}")

    # Count samples per class
    class_counts = defaultdict(int)
    for label in manner_labels:
        class_counts[label] += 1
    print("\nSamples per manner class:")
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls}: {count}")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each manner group
    for group in ['Vowel', 'Stop', 'Fricative', 'Nasal', 'Approximant']:
        mask = np.array([l == group for l in manner_labels])
        if not any(mask):
            continue
        ax.scatter(
            pca_coords[mask, 0],
            pca_coords[mask, 1],
            c=GROUP_COLORS[group],
            label=group,
            alpha=0.5,
            s=15,
            edgecolors='none'
        )

    # Add centroid labels
    for group in MANNER_GROUPS.keys():
        mask = np.array([l == group for l in manner_labels])
        if not any(mask):
            continue
        centroid = pca_coords[mask].mean(axis=0)
        ax.annotate(
            group.upper(),
            xy=centroid,
            fontsize=10,
            fontweight='bold',
            ha='center',
            va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8)
        )

    # Labels and formatting
    ax.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}% variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}% variance)', fontsize=12)
    ax.set_title('Neural Feature Space: Hidden States by Manner of Articulation', fontsize=14)

    # Legend
    ax.legend(
        title='Manner of Articulation',
        loc='upper right',
        framealpha=0.9,
        fontsize=10
    )

    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {output_path}")

    return pca, explained_var


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Load model
    model = load_model(CHECKPOINT_PATH, device)

    # Setup data loader
    dataset = setup_data_loader(batch_size=32)

    # Extract hidden states
    hidden_states, manner_labels = extract_hidden_states(
        model, dataset, device, max_frames=MAX_FRAMES
    )

    # Run PCA and create plot
    pca, explained_var = run_pca_and_plot(hidden_states, manner_labels, OUTPUT_PATH)

    print("\nDone!")


if __name__ == '__main__':
    main()
