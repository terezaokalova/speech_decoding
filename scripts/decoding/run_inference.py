#!/usr/bin/env python3
"""
TEST SET Inference Script - Generates logits for competition test set (1,450 trials).
Saves logits batch-by-batch to avoid memory issues.
Test set does NOT have labels, so no PER/loss calculation.
"""
import sys
import os
import numpy as np
import torch
import yaml
import h5py
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence

# Add the training code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'model_training_diphone'))

from rnn_trainer import BrainToTextDecoder_Trainer

def load_test_trials(dataset_dir, sessions):
    """
    Manually load test trial information from HDF5 files.
    Returns a dictionary mapping session index to trial information.
    """
    test_trials = {}

    for day_idx, session in enumerate(sessions):
        test_file_path = os.path.join(dataset_dir, session, 'data_test.hdf5')

        if not os.path.exists(test_file_path):
            print(f"  Warning: No test file for {session}")
            test_trials[day_idx] = {'trials': [], 'session_path': test_file_path}
            continue

        # Get trial indices from the file
        with h5py.File(test_file_path, 'r') as f:
            num_trials = len(list(f.keys()))
            trial_indices = list(range(num_trials))

        test_trials[day_idx] = {
            'trials': trial_indices,
            'session_path': test_file_path
        }

    return test_trials

def create_test_batch_index(test_trials, batch_size=64):
    """
    Create batch index for test data (similar to create_batch_index_test in dataset.py).
    Each batch contains trials from a single day, up to batch_size trials.
    """
    batch_index = {}
    batch_idx = 0

    for day_idx in test_trials.keys():
        if len(test_trials[day_idx]['trials']) == 0:
            continue

        # Calculate how many batches we need for this day
        num_trials = len(test_trials[day_idx]['trials'])
        num_batches = (num_trials + batch_size - 1) // batch_size

        # Create batches for this day
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_trials)

            # Get the trial indices for this batch
            batch_trials = test_trials[day_idx]['trials'][start_idx:end_idx]

            # Add to batch_index
            batch_index[batch_idx] = {day_idx: batch_trials}
            batch_idx += 1

    return batch_index

def load_batch(batch_index_entry, test_trials, feature_subset=None):
    """
    Load a batch of test data from HDF5 files.
    Returns a dictionary with 'input_features', 'n_time_steps', 'day_indicies', etc.
    Does NOT load labels (they don't exist in test set).
    """
    batch = {
        'input_features': [],
        'n_time_steps': [],
        'day_indicies': [],
        'block_nums': [],
        'trial_nums': [],
    }

    # Iterate through each day in the batch
    for day_idx, trial_list in batch_index_entry.items():
        session_path = test_trials[day_idx]['session_path']

        # Open the HDF5 file for that day
        with h5py.File(session_path, 'r') as f:
            # For each trial in this batch
            for trial_idx in trial_list:
                try:
                    g = f[f'trial_{trial_idx:04d}']

                    # Load neural data
                    input_features = torch.from_numpy(g['input_features'][:])
                    if feature_subset:
                        input_features = input_features[:, feature_subset]

                    batch['input_features'].append(input_features)
                    batch['n_time_steps'].append(g.attrs['n_time_steps'])
                    batch['day_indicies'].append(int(day_idx))
                    batch['block_nums'].append(g.attrs['block_num'])
                    batch['trial_nums'].append(g.attrs['trial_num'])

                except Exception as e:
                    print(f'Error loading trial {trial_idx} from {session_path}: {e}')
                    continue

    # Pad data to form a cohesive batch
    batch['input_features'] = pad_sequence(batch['input_features'], batch_first=True, padding_value=0)
    batch['n_time_steps'] = torch.tensor(batch['n_time_steps'])
    batch['day_indicies'] = torch.tensor(batch['day_indicies'])
    batch['block_nums'] = torch.tensor(batch['block_nums'])
    batch['trial_nums'] = torch.tensor(batch['trial_nums'])

    return batch

def main():
    # Configuration paths
    CONFIG_PATH = '/users/okalova/speech_decoding/results/diphone_run_1/checkpoint/args.yaml'
    CHECKPOINT_PATH = '/users/okalova/speech_decoding/results/diphone_run_1/checkpoint/best_checkpoint'
    OUTPUT_DIR = '/users/okalova/speech_decoding/results/diphone_run_1/logits_TEST'

    print("=" * 70)
    print("TEST SET INFERENCE - Generating Logits for Competition (1,450 trials)")
    print("=" * 70)

    # 1. Load configuration
    print(f"\n[1/6] Loading configuration from: {CONFIG_PATH}")
    with open(CONFIG_PATH, 'r') as f:
        args = yaml.safe_load(f)
    print(f"  ✓ Config loaded successfully")
    print(f"  Model: {args['model']['n_units']} units, {args['model']['n_layers']} layers")
    print(f"  Dataset: {len(args['dataset']['sessions'])} sessions")

    # 2. Create output directory
    print(f"\n[2/6] Creating output directory: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"  ✓ Directory ready")

    # 3. Load test trial information
    print(f"\n[3/6] Loading test trial information...")
    test_trials = load_test_trials(args['dataset']['dataset_dir'], args['dataset']['sessions'])

    # Count total trials
    total_test_trials = sum(len(test_trials[d]['trials']) for d in test_trials.keys())
    print(f"  ✓ Found {total_test_trials} test trials across {len(test_trials)} sessions")

    # 4. Create batch index
    print(f"\n[4/6] Creating batch index...")
    batch_size = args['dataset']['batch_size']
    batch_index = create_test_batch_index(test_trials, batch_size=batch_size)
    print(f"  ✓ Created {len(batch_index)} batches (batch_size={batch_size})")

    # 5. Initialize model (without creating dataloaders)
    print(f"\n[5/6] Initializing model...")

    # We can't use the full Trainer because it requires creating dataloaders
    # Instead, load the model directly
    from rnn_model import GRUDecoder

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Initialize model
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

    # Compile model
    print("  Compiling model...")
    model = torch.compile(model)

    # Load checkpoint
    print(f"  Loading checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)

    # Add _orig_mod. prefix for compiled model
    state_dict = checkpoint['model_state_dict']
    compiled_dict = {}
    for k, v in state_dict.items():
        clean_k = k.replace("module.", "").replace("_orig_mod.", "")
        compiled_dict[f"_orig_mod.{clean_k}"] = v

    model.load_state_dict(compiled_dict, strict=True)
    model.to(device)
    model.eval()

    print(f"  ✓ Model loaded successfully")
    print(f"  Checkpoint PER: {checkpoint['val_PER']:.4f}")
    print(f"  Checkpoint Loss: {checkpoint['val_loss']:.4f}")

    # Get feature subset if specified
    feature_subset = args['dataset'].get('feature_subset', None)

    # 6. Run inference and save batch-by-batch
    print(f"\n[6/6] Running inference on TEST set...")
    print(f"  Saving logits to: {OUTPUT_DIR}/batch_*.npy")
    print(f"  NOTE: Test set has NO labels, so no PER/loss will be calculated")
    print("-" * 70)

    batch_count = 0
    total_trials = 0

    with torch.no_grad():
        for batch_idx in sorted(batch_index.keys()):
            # Load batch manually
            batch = load_batch(batch_index[batch_idx], test_trials, feature_subset=feature_subset)

            # Move to device
            features = batch['input_features'].to(device)
            n_time_steps = batch['n_time_steps'].to(device)
            day_indicies = batch['day_indicies'].to(device)

            # Apply data transformations (same as validation)
            with torch.autocast(device_type="cuda", enabled=args['use_amp'], dtype=torch.bfloat16):
                # Apply Gaussian smoothing (no augmentations for test)
                from data_augmentations import gauss_smooth
                transform_args = args['dataset']['data_transforms']

                if transform_args['smooth_data']:
                    features = gauss_smooth(
                        inputs=features,
                        device=device,
                        smooth_kernel_std=transform_args['smooth_kernel_std'],
                        smooth_kernel_size=transform_args['smooth_kernel_size'],
                    )

                # Get model output
                outputs = model(features, day_indicies)

                # Extract main logits (handle dict output from multi-head model)
                if isinstance(outputs, dict):
                    logits = outputs['main']
                else:
                    logits = outputs

            # Convert to float32 and move to CPU
            logits_np = logits.float().cpu().numpy()

            # Save immediately to disk
            output_path = os.path.join(OUTPUT_DIR, f'batch_{batch_count:04d}.npy')
            np.save(output_path, logits_np)

            # Update counters
            batch_size_actual = logits_np.shape[0]
            total_trials += batch_size_actual
            batch_count += 1

            # Print progress
            if batch_count % 10 == 0 or batch_count <= 5:
                print(f"  Saved batch {batch_count}: shape {logits_np.shape} -> {output_path}")

    print("-" * 70)
    print(f"\n✓ INFERENCE COMPLETE")
    print(f"  Total batches saved: {batch_count}")
    print(f"  Total trials: {total_trials}")
    print(f"  Expected trials: 1,450")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  File pattern: batch_XXXX.npy")
    print("=" * 70)

    # Create a metadata file
    metadata = {
        'num_batches': batch_count,
        'num_trials': total_trials,
        'config_path': CONFIG_PATH,
        'checkpoint_path': CHECKPOINT_PATH,
        'checkpoint_per': float(checkpoint['val_PER']),
        'checkpoint_loss': float(checkpoint['val_loss']),
        'logits_shape_info': 'Each file: [batch_size, time_steps, 1601]',
        'note': 'TEST SET - No labels available (competition data)',
        'data_type': 'Diphone model outputs (1601 classes = 40*40 + 1)'
    }

    metadata_path = os.path.join(OUTPUT_DIR, 'metadata.yaml')
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f)
    print(f"\n✓ Metadata saved to: {metadata_path}")

if __name__ == "__main__":
    main()
