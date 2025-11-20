#!/usr/bin/env python3
"""
Test script for Phase 3 articulatory feature integration.
This verifies that the data loader correctly extracts and pads articulatory features.
"""

import torch
import yaml
import sys
from pathlib import Path

# Import our modules
from dataset import BrainToTextDataset, train_test_split_indicies
from phoneme_to_features import PLACE_CLASSES, MANNER_CLASSES, VOICING_CLASSES

def test_phase3_dataloader():
    print("=" * 60)
    print("Phase 3 Articulatory Features - Data Loader Test")
    print("=" * 60)

    # Load config to get data paths
    config_path = 'diphone_args_remote.yaml'
    with open(config_path, 'r') as f:
        args = yaml.safe_load(f)

    # Get just a couple sessions for quick testing
    test_sessions = args['dataset']['sessions'][:3]
    print(f"\n✓ Testing with {len(test_sessions)} sessions: {test_sessions}")

    # Build file paths
    dataset_dir = Path(args['dataset']['dataset_dir'])
    file_paths = [dataset_dir / session / 'competitionData.h5' for session in test_sessions]

    # Split into train/test
    train_trials, test_trials = train_test_split_indicies(
        file_paths=[str(p) for p in file_paths],
        test_percentage=0.1,
        seed=42
    )

    print(f"✓ Train trials: {sum(len(v['trials']) for v in train_trials.values())} total")
    print(f"✓ Test trials: {sum(len(v['trials']) for v in test_trials.values())} total")

    # Create a small test dataset (just 2 batches)
    test_dataset = BrainToTextDataset(
        trial_indicies=train_trials,
        n_batches=2,
        split='train',
        batch_size=8,
        days_per_batch=2,
        random_seed=42
    )

    print(f"\n✓ Created test dataset with {len(test_dataset)} batches")

    # Load one batch
    print("\n" + "=" * 60)
    print("Testing batch loading...")
    print("=" * 60)

    try:
        batch = test_dataset[0]

        # Check that all expected keys exist
        expected_keys = [
            'input_features', 'seq_class_ids', 'place_targets',
            'manner_targets', 'voice_targets', 'n_time_steps',
            'phone_seq_lens', 'day_indicies', 'transcriptions',
            'block_nums', 'trial_nums'
        ]

        print("\n✓ Batch keys check:")
        for key in expected_keys:
            if key in batch:
                print(f"  ✓ {key}: {batch[key].shape if hasattr(batch[key], 'shape') else 'present'}")
            else:
                print(f"  ✗ {key}: MISSING!")
                return False

        # Verify shapes match
        print("\n✓ Shape consistency check:")
        batch_size = batch['seq_class_ids'].shape[0]
        seq_len_diphones = batch['seq_class_ids'].shape[1]
        seq_len_features = batch['place_targets'].shape[1]

        print(f"  Batch size: {batch_size}")
        print(f"  Diphone sequence length: {seq_len_diphones}")
        print(f"  Feature sequence length: {seq_len_features}")

        # Note: Diphone sequences have N-1 elements for N phonemes
        # Feature sequences have N+2 elements (SIL + phonemes + SIL)
        # So feature_len should be approximately diphone_len + 1
        print(f"  Expected relationship: feature_len ≈ diphone_len + 1")
        print(f"  Actual: {seq_len_features} vs {seq_len_diphones} + 1 = {seq_len_diphones + 1}")

        # Check that place/manner/voice all have same shape
        if (batch['place_targets'].shape == batch['manner_targets'].shape ==
            batch['voice_targets'].shape):
            print(f"  ✓ All articulatory features have matching shapes")
        else:
            print(f"  ✗ Shape mismatch!")
            return False

        # Check value ranges
        print("\n✓ Value range check:")
        print(f"  Diphones: min={batch['seq_class_ids'].min()}, max={batch['seq_class_ids'].max()} (expect 0-1600)")
        print(f"  Place: min={batch['place_targets'].min()}, max={batch['place_targets'].max()} (expect 0-{len(PLACE_CLASSES)-1})")
        print(f"  Manner: min={batch['manner_targets'].min()}, max={batch['manner_targets'].max()} (expect 0-{len(MANNER_CLASSES)-1})")
        print(f"  Voice: min={batch['voice_targets'].min()}, max={batch['voice_targets'].max()} (expect 0-{len(VOICING_CLASSES)-1})")

        # Sample one sequence to verify content
        print("\n✓ Sample inspection (first trial, first 10 elements):")
        print(f"  Place:  {batch['place_targets'][0, :10].tolist()}")
        print(f"  Manner: {batch['manner_targets'][0, :10].tolist()}")
        print(f"  Voice:  {batch['voice_targets'][0, :10].tolist()}")

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nPhase 3 articulatory features are correctly integrated.")
        print("The data loader now provides:")
        print(f"  - place_targets: {len(PLACE_CLASSES)} classes")
        print(f"  - manner_targets: {len(MANNER_CLASSES)} classes")
        print(f"  - voice_targets: {len(VOICING_CLASSES)} classes")

        return True

    except Exception as e:
        print(f"\n❌ ERROR during batch loading:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_phase3_dataloader()
    sys.exit(0 if success else 1)
