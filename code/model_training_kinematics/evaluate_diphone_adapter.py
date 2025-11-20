import torch
import numpy as np

# --- DIPHONE TO PHONEME MAPPING LOGIC ---
# We need to reverse the logic we put in dataset.py.
# Diphone Index = (Prev * 40) + Curr + 1
# Therefore: Curr = (Diphone Index - 1) % 40

def convert_diphone_logits_to_phoneme_logits(diphone_logits):
    """
    Converts diphone model output (1601 classes) to phoneme space (41 classes)
    for compatibility with existing language model and evaluation pipeline.
    
    Args:
        diphone_logits: Tensor of shape [Batch, Time, 1601]
                       Index 0: CTC blank
                       Index 1-1600: Diphones (prev_phoneme * 40 + curr_phoneme + 1)
    Returns:
        phoneme_logits: Tensor of shape [Batch, Time, 41]
                       Index 0: CTC blank  
                       Index 1-40: Phonemes
    
    Strategy:
        For each diphone that ends in phoneme P, we accumulate its probability
        into the phoneme P slot. We use max-pooling in log-space (i.e., take
        the maximum logit value) as a simple approximation.
        
        More sophisticated approach: log-sum-exp marginalization
        But max is standard for greedy/beam decoding scenarios.
    """
    # 1. Initialize output tensor (Batch, Time, 41) filled with very low probs (-inf)
    B, T, _ = diphone_logits.shape
    phoneme_logits = torch.full((B, T, 41), -1e9, device=diphone_logits.device, dtype=diphone_logits.dtype)
    
    # 2. Handle CTC Blank (Index 0)
    # The diphone blank (0) maps directly to phoneme blank (0)
    phoneme_logits[:, :, 0] = diphone_logits[:, :, 0]
    
    # 3. Map Diphones (1..1600) to Phonemes (1..40)
    # For each diphone index d, compute which phoneme it represents (the 'current' phoneme)
    # Formula: d = (prev * 40) + curr + 1
    # Solving for curr: curr = (d - 1) % 40
    # In the 41-class phoneme space, curr phoneme 0..39 maps to indices 1..40
    
    # Optimization: Vectorized approach using scatter/gather
    # For implementation clarity, we'll use a loop first
    
    for d_idx in range(1, 1601):
        # Calculate which phoneme this diphone represents (the 'current' phoneme)
        # curr ranges from 0..39, which corresponds to phoneme indices 1..40
        curr_phoneme_idx = ((d_idx - 1) % 40) + 1
        
        # Max-pool: combine probability of "ending in Phoneme P" across all diphones
        # Since we're in logit space (log prob), max is appropriate for best-path decoding
        phoneme_logits[:, :, curr_phoneme_idx] = torch.max(
            phoneme_logits[:, :, curr_phoneme_idx], 
            diphone_logits[:, :, d_idx]
        )
            
    return phoneme_logits


def convert_diphone_logits_to_phoneme_logits_logsumexp(diphone_logits):
    """
    Alternative implementation using log-sum-exp for proper probability marginalization.
    This is more theoretically correct but may be overkill for greedy decoding.
    
    Args:
        diphone_logits: Tensor of shape [Batch, Time, 1601]
    Returns:
        phoneme_logits: Tensor of shape [Batch, Time, 41]
    """
    B, T, _ = diphone_logits.shape
    phoneme_logits = torch.full((B, T, 41), -1e9, device=diphone_logits.device, dtype=diphone_logits.dtype)
    
    # CTC Blank
    phoneme_logits[:, :, 0] = diphone_logits[:, :, 0]
    
    # For each phoneme, collect all diphones that end in it
    for phoneme_idx in range(1, 41):
        # Find all diphone indices that end in this phoneme
        # curr = phoneme_idx - 1 (converting from 1-indexed to 0-indexed)
        # d = prev * 40 + curr + 1
        # For all prev in [0..39]: d ranges from (prev * 40 + curr + 1)
        
        curr = phoneme_idx - 1
        diphone_indices = [prev * 40 + curr + 1 for prev in range(40)]
        
        # Extract logits for these diphones
        diphone_logits_subset = diphone_logits[:, :, diphone_indices]  # [B, T, 40]
        
        # Log-sum-exp marginalization
        phoneme_logits[:, :, phoneme_idx] = torch.logsumexp(diphone_logits_subset, dim=2)
    
    return phoneme_logits


# Example usage and testing
if __name__ == "__main__":
    # Test the adapter
    batch_size = 2
    time_steps = 100
    
    # Create dummy diphone logits
    diphone_logits = torch.randn(batch_size, time_steps, 1601)
    
    # Test max-pooling version
    phoneme_logits_max = convert_diphone_logits_to_phoneme_logits(diphone_logits)
    print(f"Input shape: {diphone_logits.shape}")
    print(f"Output shape (max): {phoneme_logits_max.shape}")
    print(f"Output shape correct: {phoneme_logits_max.shape == (batch_size, time_steps, 41)}")
    
    # Test logsumexp version
    phoneme_logits_lse = convert_diphone_logits_to_phoneme_logits_logsumexp(diphone_logits)
    print(f"Output shape (logsumexp): {phoneme_logits_lse.shape}")
    
    # Verify blank is preserved
    blank_preserved = torch.allclose(phoneme_logits_max[:, :, 0], diphone_logits[:, :, 0])
    print(f"Blank preservation (max): {blank_preserved}")
    
    blank_preserved_lse = torch.allclose(phoneme_logits_lse[:, :, 0], diphone_logits[:, :, 0])
    print(f"Blank preservation (logsumexp): {blank_preserved_lse}")
    
    print("\nAdapter functions ready for integration with evaluate_model.py")
