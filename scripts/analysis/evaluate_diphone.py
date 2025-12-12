# Wrapper to inject Adapter Logic into standard evaluation
from evaluate_model import *
from evaluate_diphone_adapter import convert_diphone_logits_to_phoneme_logits

# Monkeypatch the model forward pass is risky.
# Instead, we intercept the 'run_model' or loop logic?
# Actually, the safest way in this codebase is to subclass/wrap the model object 
# passed to the decoder.

# But 'evaluate_model.py' is a script, not a class we can easily wrap from outside without
# modifying it.
# Let's modify 'evaluate_model.py' IN PLACE to add the adapter.
# This is dirty but guaranteed to work.
pass
