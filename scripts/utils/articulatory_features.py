# Mappings based on standard IPA/ARPABET articulation
# Used to generate auxiliary targets for Multi-Task CTC Training.

# 1. DEFINE CLASSES
# We need fixed integer indices for each feature category.

PLACE_CLASSES = [
    'bilabial', 'labiodental', 'dental', 'alveolar', 'postalveolar', 
    'palatal', 'velar', 'glottal', 'vowel_front', 'vowel_central', 'vowel_back', 'silence'
]

MANNER_CLASSES = [
    'stop', 'fricative', 'affricate', 'nasal', 'liquid', 'glide', 
    'vowel_high', 'vowel_mid', 'vowel_low', 'diphthong', 'silence'
]

VOICING_CLASSES = ['voiced', 'unvoiced', 'silence']

# Helper to get indices
def get_features(phoneme):
    # Handle edge cases like 'SIL' vs '<sil>'
    p_key = phoneme.upper()
    if p_key == '<SIL>': p_key = 'SIL'
    
    p = PHONEME_MAP.get(p_key)
    if p is None:
        # Fallback for unknown tokens (map to silence/blank)
        return (PLACE_CLASSES.index('silence'), MANNER_CLASSES.index('silence'), VOICING_CLASSES.index('silence'))
    return (
        PLACE_CLASSES.index(p['place']),
        MANNER_CLASSES.index(p['manner']),
        VOICING_CLASSES.index(p['voice'])
    )

# 2. THE MAPPING DICTIONARY
PHONEME_MAP = {
    # --- VOWELS (Manner=Height, Place=Backness) ---
    'AA': {'place': 'vowel_back',    'manner': 'vowel_low',  'voice': 'voiced'},
    'AE': {'place': 'vowel_front',   'manner': 'vowel_low',  'voice': 'voiced'},
    'AH': {'place': 'vowel_central', 'manner': 'vowel_mid',  'voice': 'voiced'},
    'AO': {'place': 'vowel_back',    'manner': 'vowel_low',  'voice': 'voiced'},
    'AW': {'place': 'vowel_back',    'manner': 'diphthong',  'voice': 'voiced'},
    'AY': {'place': 'vowel_front',   'manner': 'diphthong',  'voice': 'voiced'},
    'EH': {'place': 'vowel_front',   'manner': 'vowel_mid',  'voice': 'voiced'},
    'ER': {'place': 'vowel_central', 'manner': 'vowel_mid',  'voice': 'voiced'},
    'EY': {'place': 'vowel_front',   'manner': 'diphthong',  'voice': 'voiced'},
    'IH': {'place': 'vowel_front',   'manner': 'vowel_high', 'voice': 'voiced'},
    'IY': {'place': 'vowel_front',   'manner': 'vowel_high', 'voice': 'voiced'},
    'OW': {'place': 'vowel_back',    'manner': 'diphthong',  'voice': 'voiced'},
    'OY': {'place': 'vowel_back',    'manner': 'diphthong',  'voice': 'voiced'},
    'UH': {'place': 'vowel_back',    'manner': 'vowel_high', 'voice': 'voiced'},
    'UW': {'place': 'vowel_back',    'manner': 'vowel_high', 'voice': 'voiced'},
    
    # --- CONSONANTS ---
    'B':  {'place': 'bilabial',      'manner': 'stop',       'voice': 'voiced'},
    'CH': {'place': 'postalveolar',  'manner': 'affricate',  'voice': 'unvoiced'},
    'D':  {'place': 'alveolar',      'manner': 'stop',       'voice': 'voiced'},
    'DH': {'place': 'dental',        'manner': 'fricative',  'voice': 'voiced'},
    'F':  {'place': 'labiodental',   'manner': 'fricative',  'voice': 'unvoiced'},
    'G':  {'place': 'velar',         'manner': 'stop',       'voice': 'voiced'},
    'HH': {'place': 'glottal',       'manner': 'fricative',  'voice': 'unvoiced'},
    'JH': {'place': 'postalveolar',  'manner': 'affricate',  'voice': 'voiced'},
    'K':  {'place': 'velar',         'manner': 'stop',       'voice': 'unvoiced'},
    'L':  {'place': 'alveolar',      'manner': 'liquid',     'voice': 'voiced'},
    'M':  {'place': 'bilabial',      'manner': 'nasal',      'voice': 'voiced'},
    'N':  {'place': 'alveolar',      'manner': 'nasal',      'voice': 'voiced'},
    'NG': {'place': 'velar',         'manner': 'nasal',      'voice': 'voiced'},
    'P':  {'place': 'bilabial',      'manner': 'stop',       'voice': 'unvoiced'},
    'R':  {'place': 'alveolar',      'manner': 'liquid',     'voice': 'voiced'},
    'S':  {'place': 'alveolar',      'manner': 'fricative',  'voice': 'unvoiced'},
    'SH': {'place': 'postalveolar',  'manner': 'fricative',  'voice': 'unvoiced'},
    'T':  {'place': 'alveolar',      'manner': 'stop',       'voice': 'unvoiced'},
    'TH': {'place': 'dental',        'manner': 'fricative',  'voice': 'unvoiced'},
    'V':  {'place': 'labiodental',   'manner': 'fricative',  'voice': 'voiced'},
    'W':  {'place': 'bilabial',      'manner': 'glide',      'voice': 'voiced'},
    'Y':  {'place': 'palatal',       'manner': 'glide',      'voice': 'voiced'},
    'Z':  {'place': 'alveolar',      'manner': 'fricative',  'voice': 'voiced'},
    'ZH': {'place': 'postalveolar',  'manner': 'fricative',  'voice': 'voiced'},
    
    # --- SPECIAL ---
    '<sil>': {'place': 'silence', 'manner': 'silence', 'voice': 'silence'},
    'SIL':   {'place': 'silence', 'manner': 'silence', 'voice': 'silence'},
    'SP':    {'place': 'silence', 'manner': 'silence', 'voice': 'silence'}
}
