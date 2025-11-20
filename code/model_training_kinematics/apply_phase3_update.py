#!/usr/bin/env python3
"""
Phase 3: Add Articulatory Feature Targets to Dataset
This script modifies dataset.py to extract and pad place/manner/voicing features.
"""

import re

def apply_phase3_update():
    filepath = 'dataset.py'

    with open(filepath, 'r') as f:
        content = f.read()

    print(f"Read {len(content)} bytes from {filepath}")

    # ========== STEP 1: Add imports ==========
    if "from phoneme_to_features import get_features" not in content:
        # Find the import section (after existing imports)
        import_pattern = r'(from torch\.nn\.utils\.rnn import pad_sequence\nimport math)'
        import_replacement = r'\1\nfrom phoneme_to_features import get_features\nfrom phoneme_vocab import base_idx_to_phoneme'
        content = re.sub(import_pattern, import_replacement, content)
        print("✓ Added imports for phoneme_to_features and phoneme_vocab")
    else:
        print("✓ Imports already present")

    # ========== STEP 2: Initialize articulatory feature lists in batch dict ==========
    # Add to the batch initialization dictionary
    batch_init_pattern = r"(batch = \{\n\s+'input_features' : \[\],\n\s+'seq_class_ids' : \[\],)"
    batch_init_replacement = r"\1\n            'place_targets' : [],\n            'manner_targets' : [],\n            'voice_targets' : [],"

    if "'place_targets'" not in content:
        content = re.sub(batch_init_pattern, batch_init_replacement, content)
        print("✓ Added articulatory feature lists to batch dict")
    else:
        print("✓ Articulatory feature lists already in batch dict")

    # ========== STEP 3: Extract features per-trial inside the diphone loop ==========
    # We need to add feature extraction INSIDE the existing for-loop that processes diphones
    # Find the location right after padded_base_indices is created

    feature_extraction_code = """
            # --- PHASE 3: Extract Articulatory Features ---
            # Map each base phoneme index to its articulatory features
            place_seq = []
            manner_seq = []
            voice_seq = []

            for base_idx in padded_base_indices:
                # Convert base index (0-39) to phoneme string
                phoneme = base_idx_to_phoneme(base_idx)

                # Get articulatory features (returns tuple of indices)
                p, m, v = get_features(phoneme)
                place_seq.append(p)
                manner_seq.append(m)
                voice_seq.append(v)

            # Add to batch (as Python lists for now, will be padded later)
            batch['place_targets'].append(place_seq)
            batch['manner_targets'].append(manner_seq)
            batch['voice_targets'].append(voice_seq)
            # --- END PHASE 3 ---
"""

    if "PHASE 3: Extract Articulatory Features" not in content:
        # Insert after the diphone_seq.append line and before the diphone_sequences.append
        pattern = r'(diphone_seq\.append\(diphone_idx\)\n\n)(            diphone_sequences\.append)'
        replacement = feature_extraction_code + r'\n\2'
        content = re.sub(pattern, replacement, content)
        print("✓ Added articulatory feature extraction logic")
    else:
        print("✓ Articulatory feature extraction already present")

    # ========== STEP 4: Pad articulatory features ==========
    # Add padding logic after the diphone sequences are padded
    padding_code = """
        # Pad articulatory feature sequences (Phase 3)
        batch['place_targets'] = pad_sequence(
            [torch.tensor(seq, dtype=torch.long) for seq in batch['place_targets']],
            batch_first=True,
            padding_value=0
        )
        batch['manner_targets'] = pad_sequence(
            [torch.tensor(seq, dtype=torch.long) for seq in batch['manner_targets']],
            batch_first=True,
            padding_value=0
        )
        batch['voice_targets'] = pad_sequence(
            [torch.tensor(seq, dtype=torch.long) for seq in batch['voice_targets']],
            batch_first=True,
            padding_value=0
        )
"""

    if "Pad articulatory feature sequences" not in content:
        # Insert after phone_seq_lens is converted to tensor
        pattern = r"(batch\['phone_seq_lens'\] = torch\.tensor\(diphone_seq_lens\)\n        # --- END DIPHONE MODIFICATION ---)"
        replacement = r"\1" + padding_code
        content = re.sub(pattern, replacement, content)
        print("✓ Added padding for articulatory features")
    else:
        print("✓ Padding logic already present")

    # ========== STEP 5: Write modified content ==========
    with open(filepath, 'w') as f:
        f.write(content)

    print("\n✅ Phase 3 update complete!")
    print("\nNext steps:")
    print("1. Verify syntax: python -m py_compile dataset.py")
    print("2. Test data loading: Run a quick test batch")

if __name__ == "__main__":
    apply_phase3_update()
