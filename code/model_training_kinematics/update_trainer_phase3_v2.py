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

# Inject the helper method into the class
if "def compute_multitask_loss" not in content:
    # Insert before 'def train' method
    match_method = re.search(r"(\s+)def train\(self\):", content)
    if match_method:
        indent = match_method.group(1)
        # Fix indentation of our helper
        helper_method = helper_method.replace("\n    ", "\n" + indent)
        
        # Insert before train
        content = content.replace(f"{indent}def train(self):", f"{helper_method}\n\n{indent}def train(self):")
        print("✅ Injected 'compute_multitask_loss' helper method.")
    else:
        print("❌ Error: Could not find 'def train(self):'.")


# --- 2. REPLACE THE LOSS CALCULATION ---
# This pattern should match the entire loss calculation block
loss_pattern = r"(loss\s*=\s*self\.ctc_loss\s*\(\s*log_probs\s*=\s*torch\.permute\(logits\.log_softmax\(2\),\s*\[1,\s*0,\s*2\]\),\s*targets\s*=\s*labels,\s*input_lengths\s*=\s*adjusted_lens,\s*target_lengths\s*=\s*phone_seq_lens\s*\)\s*loss\s*=\s*torch\.mean\(loss\))"

match_loss = re.search(loss_pattern, content)

if match_loss:
    original_block = match_loss.group(1)
    print(f"Found Loss Block (length={len(original_block)} chars)")
    
    # Build replacement with proper indentation
    replacement_block = """# --- PHASE 3 MULTI-TASK UPDATE ---
                if isinstance(logits, dict):
                    loss = self.compute_multitask_loss(logits, batch, adjusted_lens)
                else:
                    # Legacy (Single Head)
                    """ + original_block + """
                # ---------------------------------"""
    
    content = content.replace(original_block, replacement_block)
    print("✅ Replaced Loss Calculation logic.")
    
else:
    print("❌ CRITICAL: Could not match the Loss Calculation block via regex.")
    # Try a simpler pattern
    simple_pattern = r"loss\s*=\s*self\.ctc_loss\s*\("
    if re.search(simple_pattern, content):
        print("   Found 'loss = self.ctc_loss(' - checking manual intervention needed.")


# --- 3. FIX VALIDATION ---
# Find 'def validation' and add dict handling after model call
val_match = re.search(r"def validation\(", content)
if val_match:
    start = val_match.end()
    post_val = content[start:]
    
    # Look for model call after validation definition
    m_call = re.search(r"(logits\s*=\s*self\.model\([^)]+\))", post_val)
    
    if m_call:
        original_call = m_call.group(1)
        val_replacement = original_call + "\n                    if isinstance(logits, dict): logits = logits['main']"
        
        pre_val = content[:start]
        post_val = post_val.replace(original_call, val_replacement, 1)
        
        content = pre_val + post_val
        print("✅ Updated Validation Loop to unpack 'main' logits.")
    else:
        print("Warning: Could not find model call in validation loop.")
else:
    print("Warning: Could not find 'def validation'.")

with open(filepath, 'w') as f:
    f.write(content)

print(f"\n📝 Wrote {len(content)} bytes to {filepath}")
