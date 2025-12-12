import pickle
import os
import sys

def analyze():
    print("### STANDALONE METRICS ANALYSIS (Seed 1) ###")
    # Path to the metrics file (We hope this wasn't the one we deleted! 
    # If we deleted the big PKL, we rely on the 'clean_metrics.json' we made earlier?)
    
    # Check for JSON first (The one we created before the crash)
    json_path = "/users/okalova/speech_decoding/results/diphone_run_1/checkpoint/clean_metrics.json"
    pkl_path = "/users/okalova/speech_decoding/results/diphone_run_1/checkpoint/val_metrics.pkl"
    
    data = None
    
    if os.path.exists(json_path):
        print(f"Found Clean JSON: {json_path}")
        import json
        with open(json_path, 'r') as f:
            data = json.load(f)
        # JSON format might be different, let's handle it
        if 'avg_PER' in data:
            print(f"FINAL RESULT (PER): {data['avg_PER'] * 100:.2f}%" if data['avg_PER'] < 1.0 else f"FINAL RESULT (PER): {data['avg_PER']:.2f}%")
            return

    if os.path.exists(pkl_path):
        print(f"Found Pickle: {pkl_path}")
        with open(pkl_path, 'rb') as f:
            d = pickle.load(f)
            # Look for val_per history
            if 'val_per' in d:
                best = min(d['val_per'])
                print(f"FINAL RESULT (PER): {best * 100:.2f}%" if best < 1.0 else f"FINAL RESULT (PER): {best:.2f}%")
                return
            elif 'avg_PER' in d:
                 print(f"FINAL RESULT (PER): {d['avg_PER']}")
                 return

    print("❌ Could not find metrics file. (Did we delete it to save space?)")
    print("If so, rely on the number '16.36%' which we recorded in the chat logs.")

if __name__ == "__main__":
    analyze()
