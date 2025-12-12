#!/usr/bin/env python3
"""
Voice Head Analysis - Analyze kinematics model's voice_head to distinguish silent vs vocalized blocks.
"""
import sys
import os
import numpy as np
import torch
import yaml
import csv
from pathlib import Path

# Add the training code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'model_training_kinematics'))

from rnn_trainer_gated import BrainToTextDecoder_Trainer

def main():
    # Configuration paths
    CONFIG_PATH = '/users/okalova/speech_decoding/results/kinematics_run_1/checkpoint/args.yaml'
    CHECKPOINT_PATH = '/users/okalova/speech_decoding/results/kinematics_run_1/checkpoint/best_checkpoint'
    OUTPUT_CSV = '/users/okalova/speech_decoding/voice_analysis.csv'

    print("=" * 70)
    print("VOICE HEAD ANALYSIS - Silent vs Vocalized Block Detection")
    print("=" * 70)

    # 1. Load configuration
    print(f"\n[1/4] Loading configuration from: {CONFIG_PATH}")
    with open(CONFIG_PATH, 'r') as f:
        args = yaml.safe_load(f)
    print(f"  ✓ Config loaded successfully")

    # 2. Initialize Trainer
    print(f"\n[2/4] Initializing BrainToTextDecoder_Trainer...")
    trainer = BrainToTextDecoder_Trainer(args)
    print(f"  ✓ Trainer initialized")
    print(f"  Device: {trainer.device}")

    # 3. Load checkpoint weights
    print(f"\n[3/4] Loading checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=trainer.device, weights_only=False)

    # Add _orig_mod. prefix to all keys for compiled model
    state_dict = checkpoint['model_state_dict']
    compiled_dict = {}
    for k, v in state_dict.items():
        clean_k = k.replace("module.", "").replace("_orig_mod.", "")
        compiled_dict[f"_orig_mod.{clean_k}"] = v

    trainer.model.load_state_dict(compiled_dict, strict=True)
    trainer.model.eval()

    print(f"  ✓ Checkpoint loaded successfully")
    print(f"  Validation PER: {checkpoint['val_PER']:.4f}")

    # 4. Analyze voice head
    print(f"\n[4/4] Analyzing voice head on validation set...")
    print(f"  Saving results to: {OUTPUT_CSV}")
    print("-" * 70)

    # Open CSV file for writing
    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['batch_idx', 'trial_idx', 'mean_voicing', 'mean_silence'])

        batch_count = 0
        total_trials = 0

        with torch.no_grad():
            for i, batch in enumerate(trainer.val_loader):
                # Get batch data
                features = batch['input_features'].to(trainer.device)
                day_indicies = batch['day_indicies'].to(trainer.device)
                day = day_indicies[0].item()

                # Skip days not in validation set
                if args['dataset']['dataset_probability_val'][day] == 0:
                    continue

                # Apply data transformations
                n_time_steps = batch['n_time_steps'].to(trainer.device)

                with torch.autocast(device_type="cuda", enabled=args['use_amp'], dtype=torch.bfloat16):
                    features, n_time_steps = trainer.transform_data(features, n_time_steps, 'val')

                    # Get model output (should be a dict with 'main' and 'voice' keys)
                    outputs = trainer.model(features, day_indicies)

                    # Extract voice logits
                    if isinstance(outputs, dict) and 'voice' in outputs:
                        voice_logits = outputs['voice']
                    else:
                        print(f"  ✗ ERROR: Model does not have 'voice' head!")
                        print(f"  Output type: {type(outputs)}")
                        if isinstance(outputs, dict):
                            print(f"  Available keys: {outputs.keys()}")
                        return

                # Convert to probabilities using softmax
                # voice_logits shape: [B, T, 3]
                voice_logits = voice_logits.float()
                voice_probs = torch.softmax(voice_logits, dim=-1).cpu().numpy()

                # Process each trial in the batch
                batch_size = voice_probs.shape[0]
                for trial_idx in range(batch_size):
                    trial_probs = voice_probs[trial_idx]  # Shape: [T, 3]

                    # Calculate mean probabilities
                    # Assuming classes are: [Voiced, Unvoiced, Silence] or similar
                    # We'll calculate:
                    # - mean_voicing: average of (class_0 + class_1)
                    # - mean_silence: average of class_2

                    mean_voiced = float(np.mean(trial_probs[:, 0]))
                    mean_unvoiced = float(np.mean(trial_probs[:, 1]))
                    mean_silence = float(np.mean(trial_probs[:, 2]))
                    mean_voicing = mean_voiced + mean_unvoiced

                    # Write to CSV
                    writer.writerow([batch_count, trial_idx, mean_voicing, mean_silence])

                batch_count += 1
                total_trials += batch_size

                # Print progress
                if batch_count % 10 == 0 or batch_count <= 5:
                    print(f"  Processed batch {batch_count}, trials: {total_trials}")

    print("-" * 70)
    print(f"\n✓ ANALYSIS COMPLETE")
    print(f"  Total batches processed: {batch_count}")
    print(f"  Total trials analyzed: {total_trials}")
    print(f"  Output file: {OUTPUT_CSV}")
    print("=" * 70)

if __name__ == "__main__":
    main()
