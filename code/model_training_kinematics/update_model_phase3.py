import re
import sys
import os

# Target the file in the CURRENT directory (kinematics)
filepath = 'rnn_model.py'

if not os.path.exists(filepath):
    print(f"Error: Could not find {filepath}")
    sys.exit(1)

with open(filepath, 'r') as f:
    content = f.read()
print(f"Read {len(content)} bytes from {filepath}")

# --- A. ADD HEADS TO __INIT__ ---
# Find the self.out line and add auxiliary heads after it
init_pattern = r"(self\.out = nn\.Linear\(self\.n_units, self\.n_classes\)\s+nn\.init\.xavier_uniform_\(self\.out\.weight\))"

new_heads = r"""\1

        # --- PHASE 3: AUXILIARY ARTICULATORY HEADS ---
        # Dimensions derived from phoneme_vocab.py (Place=12, Manner=11, Voice=3)
        self.place_head = nn.Linear(self.n_units, 12)
        self.manner_head = nn.Linear(self.n_units, 11)
        self.voice_head = nn.Linear(self.n_units, 3)
        nn.init.xavier_uniform_(self.place_head.weight)
        nn.init.xavier_uniform_(self.manner_head.weight)
        nn.init.xavier_uniform_(self.voice_head.weight)
        # -------------------------------------------"""

if "self.place_head" not in content:
    content, count = re.subn(init_pattern, new_heads, content, count=1)
    if count > 0:
        print("✅ Added Auxiliary Heads to __init__.")
    else:
        print("❌ CRITICAL: Could not find self.out initialization. Manual check required.")
        print("Looking for pattern around line 82-83...")
else:
    print("ℹ️  Auxiliary Heads already present.")


# --- B. UPDATE FORWARD PASS ---
# Current structure:
#   logits = self.out(output)
#   if return_state:
#       return logits, hidden_states
#   return logits

# We need to replace this entire block with multi-head output

forward_pattern = r"""        # Compute logits
        logits = self\.out\(output\)

        if return_state:
            return logits, hidden_states

        return logits"""

new_forward_block = """        # Compute logits
        # --- PHASE 3: MULTI-HEAD FORWARD ---
        diphone_logits = self.out(output)
        place_logits = self.place_head(output)
        manner_logits = self.manner_head(output)
        voice_logits = self.voice_head(output)

        # Return dictionary for Phase 3 Trainer
        output_dict = {
            'main': diphone_logits,
            'place': place_logits,
            'manner': manner_logits,
            'voice': voice_logits
        }

        if return_state:
            return output_dict, hidden_states

        return output_dict
        # -----------------------------------"""

if "output_dict" not in content:
    if forward_pattern in content:
        content = content.replace(forward_pattern, new_forward_block)
        print("✅ Updated Forward Pass to return Dictionary.")
    else:
        print("❌ CRITICAL: Could not match exact forward pass pattern.")
        print("Attempting partial match...")

        # Try a more flexible regex
        partial_pattern = r"(logits = self\.out\(output\).*?return logits)"
        match = re.search(partial_pattern, content, re.DOTALL)
        if match:
            print(f"Found potential match at position {match.start()}-{match.end()}")
            print("Matched text:", match.group(0)[:100])
        else:
            print("Could not find any matching pattern. Please check manually.")
else:
    print("ℹ️  Forward pass already updated.")

with open(filepath, 'w') as f:
    f.write(content)

print("\n" + "="*60)
print("Update complete. Run: python -m py_compile rnn_model.py")
print("="*60)
