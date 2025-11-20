# Phase 3: Multi-Head Model Architecture - COMPLETE ✅

## Summary

Successfully upgraded the GRUDecoder model architecture to include **auxiliary prediction heads** for articulatory features (Place, Manner, Voicing). The model now outputs a dictionary with logits for all four prediction tasks.

---

## What Was Changed

### File: `rnn_model.py`

#### 1. **Added Auxiliary Heads in `__init__`** (Lines 85-93)

```python
# --- PHASE 3: AUXILIARY ARTICULATORY HEADS ---
# Dimensions derived from phoneme_vocab.py (Place=12, Manner=11, Voice=3)
self.place_head = nn.Linear(self.n_units, 12)
self.manner_head = nn.Linear(self.n_units, 11)
self.voice_head = nn.Linear(self.n_units, 3)
nn.init.xavier_uniform_(self.place_head.weight)
nn.init.xavier_uniform_(self.manner_head.weight)
nn.init.xavier_uniform_(self.voice_head.weight)
# -------------------------------------------
```

**Details:**
- Added three new linear layers parallel to the main diphone prediction head
- Each head projects from GRU hidden units (768) to class-specific dimensions
- Xavier uniform initialization for all weights (matching main head)

---

#### 2. **Modified `forward()` Method** (Lines 139-157)

**Before (Single Output):**
```python
logits = self.out(output)

if return_state:
    return logits, hidden_states

return logits
```

**After (Multi-Head Dictionary Output):**
```python
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
# -----------------------------------
```

**Key Changes:**
- All four prediction heads now run in parallel on the same GRU output
- Returns a dictionary instead of a single tensor
- Backward compatible with `return_state=True` parameter
- Dictionary keys: `'main'`, `'place'`, `'manner'`, `'voice'`

---

## Model Output Structure

### Standard Forward Pass
```python
output = model(x, day_idx)
# Returns: dict with keys ['main', 'place', 'manner', 'voice']
```

| Key | Shape | Description |
|-----|-------|-------------|
| `'main'` | `[B, T, 1601]` | Diphone logits (40×40+1 classes) |
| `'place'` | `[B, T, 12]` | Place of articulation logits |
| `'manner'` | `[B, T, 11]` | Manner of articulation logits |
| `'voice'` | `[B, T, 3]` | Voicing logits |

Where:
- `B` = batch size
- `T` = sequence length (time steps)

---

### With Hidden States
```python
output, hidden = model(x, day_idx, return_state=True)
# output: dict with four keys
# hidden: torch.Size([n_layers, B, n_units])
```

---

## Verification Tests

### ✅ Syntax Check
```bash
python -m py_compile rnn_model.py
```
**Result**: PASSED

### ✅ Model Instantiation Test
```python
model = GRUDecoder(neural_dim=512, n_units=768, n_days=10, n_classes=1601, ...)
```
**Result**:
- Main output head: 1601 classes (diphones)
- Place head: 12 classes
- Manner head: 11 classes
- Voice head: 3 classes

### ✅ Forward Pass Test
**Input**: `[4, 100, 512]` (batch=4, seq_len=100, features=512)

**Output Shapes**:
- ✓ `'main'`: `(4, 100, 1601)` ✓
- ✓ `'place'`: `(4, 100, 12)` ✓
- ✓ `'manner'`: `(4, 100, 11)` ✓
- ✓ `'voice'`: `(4, 100, 3)` ✓

**Result**: ✅ ALL TESTS PASSED

---

## Architecture Overview

```
                     ┌─────────────────┐
                     │  Neural Input   │
                     │   [B, T, 512]   │
                     └────────┬────────┘
                              │
                     ┌────────▼────────┐
                     │  Day-Specific   │
                     │  Input Layers   │
                     └────────┬────────┘
                              │
                     ┌────────▼────────┐
                     │    GRU Layers   │
                     │  (5 layers, 768 │
                     │   hidden units) │
                     └────────┬────────┘
                              │
                              │ GRU Output [B, T, 768]
                ┌─────────────┼─────────────┐
                │             │             │
       ┌────────▼───┐  ┌──────▼───┐  ┌─────▼──────┐
       │ Main Head  │  │Place Head│  │Manner Head │ ...
       │  (1601)    │  │   (12)   │  │   (11)     │
       └────────┬───┘  └──────┬───┘  └─────┬──────┘
                │             │             │
                └─────────────┼─────────────┘
                              │
                     ┌────────▼────────┐
                     │  Output Dict    │
                     │ {main, place,   │
                     │  manner, voice} │
                     └─────────────────┘
```

---

## Model Parameter Count

**Additional Parameters Added:**
- Place head: `768 × 12 + 12 = 9,228`
- Manner head: `768 × 11 + 11 = 8,459`
- Voice head: `768 × 3 + 3 = 2,307`
- **Total new params: ~20K** (negligible compared to main model)

---

## Next Steps

Now that the model outputs multi-head logits, you need to update the **trainer** (`rnn_trainer.py`):

1. **Modify loss computation** to handle dictionary output:
   ```python
   # Old:
   loss = ctc_loss(logits, targets, ...)

   # New:
   main_loss = ctc_loss(logits['main'], batch['seq_class_ids'], ...)
   place_loss = ctc_loss(logits['place'], batch['place_targets'], ...)
   manner_loss = ctc_loss(logits['manner'], batch['manner_targets'], ...)
   voice_loss = ctc_loss(logits['voice'], batch['voice_targets'], ...)

   total_loss = main_loss + α*place_loss + β*manner_loss + γ*voice_loss
   ```

2. **Update evaluation code** to extract `logits['main']` for decoding

3. **Add loss weighting hyperparameters** (α, β, γ) to config

---

## Workspace Location

**Updated files are in:**
```
/users/okalova/speech_decoding/code/model_training_kinematics/
```

This is a **safe copy** of `model_training_diphone/` with Phase 3 model changes.

---

## Status: ✅ MODEL ARCHITECTURE READY

The multi-head model is complete and tested. Next phase: Update the trainer to compute multi-task losses.

---

*Generated: 2025-11-19*
*Phase 3 Model Architecture Complete*
