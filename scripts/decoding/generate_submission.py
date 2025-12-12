#!/usr/bin/env python3
"""
Generate Kaggle submission.csv from Phase 2 Diphone Model (Seed 1).

This script:
1. Loads the best checkpoint from results/diphone_run_1
2. Loads all test data (data_test.hdf5) from each session
3. Runs neural data through the RNN to get diphone logits
4. Converts diphone logits -> phoneme logits for LM compatibility
5. Sends logits to Redis-based Language Model for text decoding
6. Outputs submission.csv in required format: id,text

Expected test sentences: 1,450
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import redis
import time
from tqdm import tqdm
from omegaconf import OmegaConf
import argparse

# Add model_training_diphone to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DIPHONE_CODE_DIR = os.path.join(PROJECT_ROOT, 'code', 'model_training_diphone')
sys.path.insert(0, DIPHONE_CODE_DIR)

from rnn_model import GRUDecoder
from evaluate_model_helpers import (
    load_h5py_file,
    runSingleDecodingStep,
    rearrange_speech_logits_pt,
    get_current_redis_time_ms,
    reset_remote_language_model,
    send_logits_to_remote_lm,
    finalize_remote_lm,
    update_remote_lm_params,
)
from evaluate_diphone_adapter import convert_diphone_logits_to_phoneme_logits


def parse_args():
    parser = argparse.ArgumentParser(description='Generate Kaggle submission from Phase 2 Diphone model')
    parser.add_argument('--model_path', type=str,
                        default=os.path.join(PROJECT_ROOT, 'results', 'diphone_run_1'),
                        help='Path to the trained model directory')
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(PROJECT_ROOT, 'data'),
                        help='Path to the dataset directory')
    parser.add_argument('--output_file', type=str,
                        default=os.path.join(PROJECT_ROOT, 'submission.csv'),
                        help='Path for output submission.csv')
    parser.add_argument('--gpu_number', type=int, default=0,
                        help='GPU number to use. Set to -1 for CPU.')
    parser.add_argument('--redis_host', type=str, default='localhost',
                        help='Redis server host')
    parser.add_argument('--redis_port', type=int, default=6379,
                        help='Redis server port')
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("KAGGLE SUBMISSION GENERATOR - Phase 2 Diphone Model")
    print("=" * 60)

    # Load model configuration
    model_args_path = os.path.join(args.model_path, 'checkpoint', 'args.yaml')
    print(f"\nLoading model config from: {model_args_path}")
    model_args = OmegaConf.load(model_args_path)

    # Setup device
    if torch.cuda.is_available() and args.gpu_number >= 0:
        if args.gpu_number >= torch.cuda.device_count():
            print(f"Warning: GPU {args.gpu_number} not available, using GPU 0")
            args.gpu_number = 0
        device = torch.device(f'cuda:{args.gpu_number}')
        print(f"Using GPU: {device}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # Initialize model
    print("\nInitializing GRU Decoder model...")
    model = GRUDecoder(
        neural_dim=model_args['model']['n_input_features'],
        n_units=model_args['model']['n_units'],
        n_days=len(model_args['dataset']['sessions']),
        n_classes=model_args['dataset']['n_classes'],
        rnn_dropout=model_args['model']['rnn_dropout'],
        input_dropout=model_args['model']['input_network']['input_layer_dropout'],
        n_layers=model_args['model']['n_layers'],
        patch_size=model_args['model']['patch_size'],
        patch_stride=model_args['model']['patch_stride'],
    )

    # Load checkpoint
    checkpoint_path = os.path.join(args.model_path, 'checkpoint', 'best_checkpoint')
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)

    # Handle DataParallel/compiled model key naming
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "").replace("_orig_mod.", "")
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    print("Model loaded successfully!")

    # Load test data for each session
    print("\n" + "=" * 60)
    print("LOADING TEST DATA")
    print("=" * 60)

    sessions = model_args['dataset']['sessions']
    test_data = {}
    total_test_trials = 0

    for session in sessions:
        test_file = os.path.join(args.data_dir, session, 'data_test.hdf5')
        if os.path.exists(test_file):
            data = load_h5py_file(test_file, None)
            test_data[session] = data
            n_trials = len(data['neural_features'])
            total_test_trials += n_trials
            print(f"  {session}: {n_trials} test trials")

    print(f"\nTotal test trials: {total_test_trials}")
    if total_test_trials != 1450:
        print(f"WARNING: Expected 1,450 test trials, got {total_test_trials}")

    # Generate logits for all test trials
    print("\n" + "=" * 60)
    print("GENERATING NEURAL LOGITS")
    print("=" * 60)

    with tqdm(total=total_test_trials, desc='Running RNN inference', unit='trial') as pbar:
        for session, data in test_data.items():
            data['logits'] = []
            input_layer = sessions.index(session)

            for trial_idx in range(len(data['neural_features'])):
                neural_input = data['neural_features'][trial_idx]
                neural_input = np.expand_dims(neural_input, axis=0)
                neural_input = torch.tensor(neural_input, device=device, dtype=torch.bfloat16)

                # Run decoding
                logits = runSingleDecodingStep(neural_input, input_layer, model, model_args, device)

                # Convert 1601 diphone logits -> 41 phoneme logits
                logits_tensor = torch.tensor(logits, device=device, dtype=torch.float32)
                phoneme_logits = convert_diphone_logits_to_phoneme_logits(logits_tensor).cpu().numpy()

                data['logits'].append(phoneme_logits)
                pbar.update(1)

    # Connect to Redis LM server
    print("\n" + "=" * 60)
    print("CONNECTING TO LANGUAGE MODEL (Redis)")
    print("=" * 60)

    try:
        r = redis.Redis(host=args.redis_host, port=args.redis_port, db=0)
        r.ping()
        print(f"Connected to Redis at {args.redis_host}:{args.redis_port}")
    except redis.ConnectionError:
        print("ERROR: Cannot connect to Redis server!")
        print("Please start the LM server first:")
        print(f"  conda activate b2txt25_lm")
        print(f"  python code/language_model/language-model-standalone.py")
        sys.exit(1)

    r.flushall()  # Clear streams

    # Initialize Redis stream timestamps
    remote_lm_output_partial_lastEntrySeen = get_current_redis_time_ms(r)
    remote_lm_output_final_lastEntrySeen = get_current_redis_time_ms(r)
    remote_lm_done_resetting_lastEntrySeen = get_current_redis_time_ms(r)

    remote_lm_input_stream = 'remote_lm_input'
    remote_lm_output_partial_stream = 'remote_lm_output_partial'
    remote_lm_output_final_stream = 'remote_lm_output_final'

    # ========== FORCE OPTIMAL LM PARAMETERS ==========
    # Best config from grid search: acoustic_scale=1.0, blank_penalty=90.0
    print("\n" + "=" * 60)
    print("SETTING LM PARAMETERS (OPTIMIZED)")
    print("=" * 60)
    print("  acoustic_scale = 1.0")
    print("  blank_penalty  = 90.0")
    print("  alpha          = 0.55")

    remote_lm_done_updating_lastEntrySeen = get_current_redis_time_ms(r)
    remote_lm_done_updating_lastEntrySeen = update_remote_lm_params(
        r,
        remote_lm_done_updating_lastEntrySeen,
        acoustic_scale=1.0,
        blank_penalty=90.0,
        alpha=0.55,
    )
    print("  Parameters updated successfully!")
    # =================================================

    # Decode all trials through LM
    print("\n" + "=" * 60)
    print("DECODING WITH LANGUAGE MODEL")
    print("=" * 60)

    results = []

    with tqdm(total=total_test_trials, desc='LM decoding', unit='trial') as pbar:
        for session in sessions:
            if session not in test_data:
                continue

            data = test_data[session]
            for trial_idx in range(len(data['logits'])):
                # Get logits and rearrange for LM
                logits = rearrange_speech_logits_pt(data['logits'][trial_idx])[0]

                # Reset LM
                remote_lm_done_resetting_lastEntrySeen = reset_remote_language_model(
                    r, remote_lm_done_resetting_lastEntrySeen
                )

                # Send logits to LM
                remote_lm_output_partial_lastEntrySeen, _ = send_logits_to_remote_lm(
                    r,
                    remote_lm_input_stream,
                    remote_lm_output_partial_stream,
                    remote_lm_output_partial_lastEntrySeen,
                    logits,
                )

                # Finalize and get result
                remote_lm_output_final_lastEntrySeen, lm_out = finalize_remote_lm(
                    r,
                    remote_lm_output_final_stream,
                    remote_lm_output_final_lastEntrySeen,
                )

                # Get best candidate
                best_sentence = lm_out['candidate_sentences'][0] if lm_out['candidate_sentences'] else ""

                results.append({
                    'session': session,
                    'block': data['block_num'][trial_idx],
                    'trial': data['trial_num'][trial_idx],
                    'text': best_sentence,
                })

                pbar.update(1)

    # Create submission DataFrame
    print("\n" + "=" * 60)
    print("GENERATING SUBMISSION FILE")
    print("=" * 60)

    # Sort by session, block, trial to ensure chronological order
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(['session', 'block', 'trial']).reset_index(drop=True)

    # Create submission format
    submission_df = pd.DataFrame({
        'id': range(len(results_df)),
        'text': results_df['text']
    })

    # Save submission
    submission_df.to_csv(args.output_file, index=False)
    print(f"\nSubmission saved to: {args.output_file}")

    # Validation output
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)
    print(f"Total sentences generated: {len(submission_df)}")
    print(f"Expected: 1,450")

    print("\nFirst 5 generated sentences:")
    print("-" * 40)
    for i in range(min(5, len(submission_df))):
        print(f"  [{i}]: {submission_df.loc[i, 'text']}")

    print("\n" + "=" * 60)
    print("SUBMISSION COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
