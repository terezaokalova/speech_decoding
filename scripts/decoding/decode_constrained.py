import torch
import torchaudio
import pickle
import numpy as np
import argparse
import os
import yaml
from typing import List

# --- 1. CONFIGURATION & MAPPINGS ---
# 40 Base Units: 0=Blank, 1-39=Phonemes, 40=Silence (Check your vocab!)
# Based on your finding: 40 base units (indices 0-39 for creating diphones)
# Plus CTC blank at index 0 of the 1601 output.
PHONEMES = [
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY",
    "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY",
    "P", "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH", "SIL"
]
# CTC output usually adds a blank at 0. So our phoneme logits will be 41 classes.
# 0 = Blank, 1..40 = Phonemes+SIL.

def load_vocab(vocab_path):
    """Parses the competition vocabulary file."""
    words = []
    with open(vocab_path, 'r') as f:
        for line in f:
            # Handle FST format "word ID" or just "word"
            parts = line.strip().split()
            if len(parts) >= 1:
                words.append(parts[0])
    return words

def parse_lexicon(cmu_path, vocab_list):
    """
    Parses the CMU Pronouncing Dictionary and creates a lexicon.

    Args:
        cmu_path: Path to cmudict.0.7a file
        vocab_list: List of words to include (competition vocabulary)

    Returns:
        Dictionary mapping words to phoneme sequences: {"WORD": ["W", "ER", "D"], ...}
    """
    vocab_set = set(w.upper() for w in vocab_list)
    lexicon = {}

    with open(cmu_path, 'r', encoding='latin-1') as f:
        for line in f:
            # Skip comments
            if line.startswith(';;;'):
                continue

            parts = line.strip().split()
            if len(parts) < 2:
                continue

            # Word is first element
            word = parts[0]

            # Handle variant pronunciations (e.g., "WORD(2)")
            if '(' in word:
                word = word.split('(')[0]

            # Only keep words in our vocabulary
            if word not in vocab_set:
                continue

            # Phonemes are remaining elements
            phonemes = parts[1:]

            # Strip stress markers (AA1 -> AA, AH0 -> AH, etc.)
            phonemes_no_stress = []
            for p in phonemes:
                # Remove trailing digits (stress markers)
                p_clean = ''.join(c for c in p if not c.isdigit())
                phonemes_no_stress.append(p_clean)

            # Keep first pronunciation if multiple exist
            if word not in lexicon:
                lexicon[word] = phonemes_no_stress

    return lexicon

def diphone_to_phoneme_logits(diphone_logits):
    """
    Converts Diphone (1601) -> Phoneme (41).
    Logic: Max-pool all diphones that end in a specific phoneme.
    Formula: d_idx = (prev * 40) + curr + 1
    Reverse: curr = ((d_idx - 1) % 40)
    """
    B, T, D = diphone_logits.shape
    # 41 classes: 0=Blank, 1..40=Phonemes
    phoneme_logits = torch.full((B, T, 41), float('-inf'))

    # 1. Copy Blank (Index 0)
    phoneme_logits[:, :, 0] = torch.from_numpy(diphone_logits[:, :, 0])

    # 2. Max Pool for Phonemes
    # We iterate 1..1600 (The diphone indices)
    d_logits_torch = torch.from_numpy(diphone_logits)

    for d_idx in range(1, 1601):
        # Your logic: curr_phoneme_idx = ((d_idx - 1) % 40) + 1
        # This maps diphone 1 -> phoneme 1, diphone 40 -> phoneme 40
        curr_p_idx = ((d_idx - 1) % 40) + 1

        phoneme_logits[:, :, curr_p_idx] = torch.maximum(
            phoneme_logits[:, :, curr_p_idx],
            d_logits_torch[:, :, d_idx]
        )

    return phoneme_logits

def build_phoneme_to_words_map(lexicon):
    """
    Creates a reverse mapping from phoneme sequences to words.

    Args:
        lexicon: Dictionary mapping words to phoneme sequences

    Returns:
        Dictionary mapping phoneme sequences (as tuples) to lists of words
    """
    phoneme_to_words = {}
    for word, phonemes in lexicon.items():
        # Convert phoneme list to tuple (hashable)
        phoneme_key = tuple(phonemes)
        if phoneme_key not in phoneme_to_words:
            phoneme_to_words[phoneme_key] = []
        phoneme_to_words[phoneme_key].append(word)
    return phoneme_to_words

def segment_phonemes_to_words(phoneme_seq, phoneme_to_words):
    """
    Greedily segments a phoneme sequence into valid words using the lexicon.

    Args:
        phoneme_seq: List of phoneme strings (e.g., ["Y", "UW", "K", "AH", "D"])
        phoneme_to_words: Map from phoneme tuples to word lists

    Returns:
        List of words found
    """
    words = []
    i = 0
    while i < len(phoneme_seq):
        # Try longest match first (up to 20 phonemes per word)
        best_match = None
        best_len = 0

        for length in range(min(20, len(phoneme_seq) - i), 0, -1):
            subseq = tuple(phoneme_seq[i:i+length])
            if subseq in phoneme_to_words:
                best_match = phoneme_to_words[subseq][0]  # Take first word
                best_len = length
                break

        if best_match:
            words.append(best_match)
            i += best_len
        else:
            # Skip this phoneme if no match
            i += 1

    return words

def run_constrained_decoding(logits, lexicon):
    """
    Greedy CTC decoding with lexicon-constrained word segmentation.

    This is a simplified alternative to beam search that:
    1. Uses greedy CTC decoding to get phoneme sequences
    2. Segments phonemes into words using the lexicon

    Args:
        logits: Phoneme logits [B, T, 41]
        lexicon: Dictionary mapping words to phoneme sequences

    Returns:
        List of decoded transcripts (one per batch sample)
    """
    print(f"  Building phoneme-to-word lookup table...")
    phoneme_to_words = build_phoneme_to_words_map(lexicon)
    print(f"  Lookup table has {len(phoneme_to_words)} unique phoneme sequences")

    # Greedy CTC decoding
    preds = torch.argmax(logits, dim=-1)

    transcripts = []
    for seq in preds:
        # Dedup consecutive and remove blank (0)
        unique_seq = torch.unique_consecutive(seq)
        clean_seq = [idx.item() for idx in unique_seq if idx != 0]

        # Convert indices to phoneme strings
        if len(clean_seq) > 0:
            phoneme_seq = [PHONEMES[idx-1] for idx in clean_seq]

            # Remove SIL tokens for word matching
            phoneme_seq_no_sil = [p for p in phoneme_seq if p != "SIL"]

            # Segment into words using lexicon
            words = segment_phonemes_to_words(phoneme_seq_no_sil, phoneme_to_words)

            transcript = " ".join(words) if words else ""
            transcripts.append(transcript)
        else:
            transcripts.append("")

    return transcripts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logits_dir', type=str, required=True, help='Path to directory with batch_*.npy files')
    parser.add_argument('--vocab_path', type=str, required=True, help='Path to words.txt')
    parser.add_argument('--cmu_dict', type=str, default='/users/okalova/speech_decoding/code/language_model/cmudict.0.7a', help='Path to CMU dictionary')
    parser.add_argument('--output_csv', type=str, default='submission_constrained.csv')
    args = parser.parse_args()

    print("=" * 70)
    print("LEXICON-CONSTRAINED BEAM SEARCH DECODER")
    print("=" * 70)

    print(f"\n[1/4] Loading logits from {args.logits_dir}...")

    # Load metadata
    metadata_path = os.path.join(args.logits_dir, 'metadata.yaml')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = yaml.safe_load(f)
        print(f"  Metadata: {metadata['num_batches']} batches, {metadata['num_trials']} trials")
        print(f"  Checkpoint PER: {metadata['checkpoint_per']:.4f}")

    # Find all batch files
    batch_files = sorted([f for f in os.listdir(args.logits_dir) if f.startswith('batch_') and f.endswith('.npy')])
    print(f"  Found {len(batch_files)} batch files")

    print(f"\n[2/4] Loading vocabulary from {args.vocab_path}...")
    vocab = load_vocab(args.vocab_path)
    print(f"  Loaded {len(vocab)} words from vocabulary")

    print(f"\n[3/4] Parsing CMU Dictionary and creating lexicon...")
    print(f"  Reading from {args.cmu_dict}...")
    lexicon = parse_lexicon(args.cmu_dict, vocab)
    print(f"  Created lexicon with {len(lexicon)} words")
    print(f"  Sample entries:")
    for i, (word, phonemes) in enumerate(list(lexicon.items())[:5]):
        print(f"    {word}: {' '.join(phonemes)}")

    print(f"\n[4/4] Running lexicon-constrained beam search decoding...")
    print(f"  Writing results to: {args.output_csv}")
    print("-" * 70)

    # Open CSV file for writing at the beginning with line buffering
    with open(args.output_csv, 'w', buffering=1) as csv_file:
        # Write header immediately
        csv_file.write("id,text\n")
        csv_file.flush()
        os.fsync(csv_file.fileno())

        trial_id = 0
        total_batches = len(batch_files)

        # Process ALL batches
        for batch_idx, batch_file in enumerate(batch_files):
            # Load batch from numpy file
            batch_path = os.path.join(args.logits_dir, batch_file)
            batch_logits = np.load(batch_path)

            print(f"\nProcessing batch {batch_idx+1}/{total_batches}: {batch_file}, Shape: {batch_logits.shape}")

            # 1. Collapse Diphones -> Phonemes
            p_logits = diphone_to_phoneme_logits(batch_logits)

            # Only print lookup table build message once
            if batch_idx == 0:
                print(f"  Converted to phoneme logits: {p_logits.shape}")

            # 2. Run lexicon-constrained beam search
            transcripts = run_constrained_decoding(p_logits, lexicon)

            # 3. Write each transcript to CSV immediately
            for transcript in transcripts:
                # Escape quotes in transcript if present
                escaped_transcript = transcript.replace('"', '""')
                csv_file.write(f"{trial_id},{escaped_transcript}\n")
                trial_id += 1

            # Force data to disk after every batch to prevent data loss
            csv_file.flush()
            os.fsync(csv_file.fileno())

            print(f"  Decoded {len(transcripts)} samples (total: {trial_id} trials written)")

    print("\n" + "=" * 70)
    print(f"✓ DECODING COMPLETE")
    print(f"  Total trials decoded: {trial_id}")
    print(f"  Output file: {args.output_csv}")
    print("=" * 70)

if __name__ == "__main__":
    main()
