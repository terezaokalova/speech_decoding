# Standard phoneme vocabulary for T15 dataset
# Based on TIMIT/ARPABET phoneme set (39 phonemes + silence)
# 
# Index mapping:
# - CTC indices: 0 = blank, 1-40 = phonemes
# - Base indices (used in dataset.py): 0-39 (shift by -1 from CTC indices)
# - Index 39 (base) / 40 (CTC) = SIL (silence)

# This is the standard ordering used in the preprocessed HDF5 files
PHONEME_LIST = [
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH',  # 0-9
    'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K',   # 10-19
    'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',      # 20-29
    'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH', 'SIL'     # 30-39
]

def base_idx_to_phoneme(idx):
    """
    Convert base index (0-39) to phoneme string.
    
    Args:
        idx: Integer index in range [0, 39]
    
    Returns:
        Phoneme string (e.g., 'AA', 'SIL')
    """
    if 0 <= idx < len(PHONEME_LIST):
        return PHONEME_LIST[idx]
    else:
        return 'SIL'  # Fallback to silence

def ctc_idx_to_phoneme(idx):
    """
    Convert CTC index (0-40) to phoneme string.
    
    Args:
        idx: Integer index in range [0, 40]
            0 = CTC blank token
            1-40 = phonemes (maps to PHONEME_LIST[0-39])
    
    Returns:
        Phoneme string or None for blank
    """
    if idx == 0:
        return None  # CTC blank
    elif 1 <= idx <= 40:
        return PHONEME_LIST[idx - 1]
    else:
        return 'SIL'  # Fallback

