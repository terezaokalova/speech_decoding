import re
import sys
import os

filepath = 'rnn_trainer.py'
if not os.path.exists(filepath):
    print(f"Error: Could not find {filepath}")
    sys.exit(1)

with open(filepath, 'r') as f:
    content = f.read()
print(f"Read {len(content)} bytes from {filepath}")

# --- 1. DEFINE THE MULTI-TASK HELPER METHOD ---
# This method handles the permutation [1,0,2], the log_softmax, and the mean reduction 
# for all 4 heads exactly as the original code did for one.

helper_method = """
    def compute_multitask_loss(self, outputs_dict, batch, input_lengths):
        # Weights
        AUX_WEIGHT = 0.2
        target_lengths = batch['phone_seq_lens']

        # 1. Main Task (Diphone)
        # Matches original: torch.permute(logits.log_softmax(2), [1, 0, 2])
        main_logits = outputs_dict['main']
        main_log_probs = torch.permute(main_logits.log_softmax(2), [1, 0, 2])
        
        loss_main = self.ctc_loss(
            log_probs=main_log_probs,
            targets=batch['seq_class_ids'],
            input_lengths=input_lengths,
            target_lengths=target_lengths
        )
        loss_main = torch.mean(loss_main)

        # 2. Aux Tasks
        # We perform the exact same ops for Place, Manner, Voice
        loss_aux = 0.0
        
        # Place
        if 'place' in outputs_dict and 'place_targets' in batch:
            place_log_probs = torch.permute(outputs_dict['place'].log_softmax(2), [1, 0, 2])
            l_place = self.ctc_loss(place_log_probs, batch['place_targets'], input_lengths, target_lengths)
            loss_aux += torch.mean(l_place)

        # Manner
        if 'manner' in outputs_dict and 'manner_targets' in batch:
            manner_log_probs = torch.permute(outputs_dict['manner'].log_softmax(2), [1, 0, 2])
            l_manner = self.ctc_loss(manner_log_probs, batch['manner_targets'], input_lengths, target_lengths)
            loss_aux += torch.mean(l_manner)

        # Voice
        if 'voice' in outputs_dict and 'voice_targets' in batch:
            voice_log_probs = torch.permute(outputs_dict['voice'].log_softmax(2), [1, 0, 2])
            l_voice = self.ctc_loss(voice_log_probs, batch['voice_targets'], input_lengths, target_lengths)
            loss_aux += torch.mean(l_voice)

        # 3. Total Loss
        total_loss = loss_main + (AUX_WEIGHT * loss_aux)
        
        return total_loss
"""

# Inject the helper method into the class (before __init__ or at the end)
if "def compute_multitask_loss" not in content:
    # We insert it before 'def train_step' or similar.
    # Let's find the class definition start to ensure indentation is correct?
    # Easier: Insert it before the 'train_batch' or 'train_step' method.
    
    # Find indentation of train methods
    match_method = re.search(r"(\s+)def train_step", content)
    if match_method:
        indent = match_method.group(1)
        # Fix indentation of our helper
        helper_method = helper_method.replace("\n    ", "\n" + indent)
        
        # Insert before train_step
        content = content.replace(f"{indent}def train_step", f"{helper_method}\n\n{indent}def train_step")
        print("✅ Injected 'compute_multitask_loss' helper method.")
    else:
        print("❌ Error: Could not find 'def train_step'.")


# --- 2. REPLACE THE LOSS CALCULATION ---
# We need to find the block where loss is calculated.
# Based on recon: 
# loss = self.ctc_loss(
#      log_probs = torch.permute(logits.log_softmax(2), [1, 0, 2]),
#      targets = labels,
#      input_lengths = adjusted_lens,
#      target_lengths = phone_seq_lens
# )
# loss = torch.mean(loss)

# We will look for the 'loss = self.ctc_loss' line.
# We replace the whole block with a conditional check.

loss_pattern = r"(loss\s*=\s*self\.ctc_loss\s*\([\s\S]*?\)\s*loss\s*=\s*torch\.mean\(loss\))"
match_loss = re.search(loss_pattern, content)

if match_loss:
    original_block = match_loss.group(1)
    print(f"Found Loss Block: {original_block[:50]}...")
    
    # The 'logits' variable holds the model output.
    # 'adjusted_lens' holds the input lengths.
    
    # Replacement Logic:
    # If logits is a dict -> use helper.
    # Else -> use original block.
    # CRITICAL: In the original block, we must ensure 'logits' is treated as tensor.
    # (The original code assumes it is).
    
    replacement_block = f"""
        # --- PHASE 3 MULTI-TASK UPDATE ---
        if isinstance(logits, dict):
            loss = self.compute_multitask_loss(logits, batch, adjusted_lens)
        else:
            # Legacy (Single Head)
            {original_block}
        # ---------------------------------
    """
    
    content = content.replace(original_block, replacement_block)
    print("✅ Replaced Loss Calculation logic.")
    
else:
    print("❌ CRITICAL: Could not match the Loss Calculation block via regex.")
    print("Please check spacing in 'loss = self.ctc_loss(...)'.")


# --- 3. FIX VALIDATION (OPTIONAL BUT SAFE) ---
# In validation, we just want 'main' logits for PER.
# Find: logits = self.model(input_features)
# Add: if isinstance(logits, dict): logits = logits['main']

# We look for the model call. It might appear multiple times.
# We want to patch it everywhere to be safe, OR specifically in validation.
# Given we fixed the Trainer to handle dicts in training, we primarily need to fix it 
# where metrics (PER) are calculated, which expects a tensor.

# Simple approach: Wrap the model call everywhere.
# Pattern: logits = self.model(input_features)
model_call = r"(logits\s*=\s*self\.model\s*\(.*?\))"
# We replace it with:
# logits = self.model(...)
# if isinstance(logits, dict) and not self.model.training: logits = logits['main']
# Actually, even in training, subsequent code (like gradient norm logging) might expect tensor?
# No, only the loss function uses logits in training.
# In validation, 'logits' is passed to decoder.
# So let's unwrap it immediately after call.

# Caveat: In training, we NEED the dict for the loss helper we just wrote.
# So we ONLY unwrap if we are NOT calculating loss.
# This is getting complex to regex safely.

# BETTER STRATEGY for VALIDATION:
# Find the specific Validation Loop model call.
# Heuristic: it's inside 'validate' method.

val_match = re.search(r"def validate\(.*?:", content)
if val_match:
    start = val_match.end()
    # Look for model call AFTER this point
    post_val = content[start:]
    m_call = re.search(model_call, post_val)
    
    if m_call:
        original_call = m_call.group(1)
        # We modify ONLY this call
        # We replace "logits = ..." with "logits = ... \n if isinstance(logits, dict): logits = logits['main']"
        
        # We need to do this string replacement carefully in the full content.
        # Find the exact string of the match
        
        # We construct a unique replacement for the validation call
        val_replacement = original_call + "\n            if isinstance(logits, dict): logits = logits['main']"
        
        # We replace the FIRST occurrence found in the validation section
        # We split content
        pre_val = content[:start]
        val_content = content[start:]
        
        val_content = val_content.replace(original_call, val_replacement, 1)
        
        content = pre_val + val_content
        print("✅ Updated Validation Loop to unpack 'main' logits.")
    else:
        print("Warning: Could not find model call in validation loop.")
else:
    print("Warning: Could not find 'def validate'.")

with open(filepath, 'w') as f:
    f.write(content)
