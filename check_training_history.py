"""
Check if the model was trained with the fixed training script.
"""

import torch
import os
from datetime import datetime

print("="*60)
print("TRAINING HISTORY CHECK")
print("="*60)

# Check model file info
model_path = 'model/model.pth'
if os.path.exists(model_path):
    stat = os.stat(model_path)
    mod_time = datetime.fromtimestamp(stat.st_mtime)
    size_mb = stat.st_size / (1024 * 1024)
    
    print(f"\nModel file: {model_path}")
    print(f"  Last modified: {mod_time}")
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  Age: {(datetime.now() - mod_time).total_seconds() / 3600:.1f} hours old")
else:
    print(f"\n❌ Model file not found: {model_path}")

# Check /tmp/model.pth
tmp_model = '/tmp/model.pth'
if os.path.exists(tmp_model):
    stat = os.stat(tmp_model)
    mod_time = datetime.fromtimestamp(stat.st_mtime)
    size_mb = stat.st_size / (1024 * 1024)
    
    print(f"\nTmp model file: {tmp_model}")
    print(f"  Last modified: {mod_time}")
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  Age: {(datetime.now() - mod_time).total_seconds() / (3600*24):.1f} days old")
    
    if (datetime.now() - mod_time).total_seconds() > 3600:  # More than 1 hour old
        print(f"  ⚠ WARNING: This is an OLD model from a previous training run!")
else:
    print(f"\n/tmp/model.pth not found (this is normal if training hasn't run recently)")

# Load and inspect state dict
print("\n" + "-"*60)
print("MODEL STATE INSPECTION")
print("-"*60)

try:
    state_dict = torch.load(model_path, map_location='cpu')
    
    # Check if there are any metadata or optimizer states
    if isinstance(state_dict, dict):
        if 'epoch' in state_dict:
            print(f"\n✓ Model contains training metadata:")
            print(f"  Epoch: {state_dict.get('epoch', 'N/A')}")
            if 'optimizer' in state_dict:
                print(f"  Has optimizer state: Yes")
        else:
            print(f"\n⚠ Model contains only weights (no training metadata)")
            print(f"  This suggests it's a simple state_dict save")
    
    # Check keys
    if isinstance(state_dict, dict):
        sample_keys = list(state_dict.keys())[:5]
        has_model_prefix = any('model.' in k for k in state_dict.keys())
        
        print(f"\nState dict structure:")
        print(f"  Total keys: {len(state_dict)}")
        print(f"  Has 'model.' prefix: {has_model_prefix}")
        print(f"  Sample keys: {sample_keys}")
        
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Check for MLflow runs
print("\n" + "-"*60)
print("MLFLOW TRAINING LOGS")
print("-"*60)

mlruns_path = 'mlruns'
if os.path.exists(mlruns_path):
    # Find recent runs
    import glob
    run_paths = glob.glob('mlruns/*/*/meta.yaml')
    if run_paths:
        # Sort by modification time
        run_paths.sort(key=os.path.getmtime, reverse=True)
        print(f"\nFound {len(run_paths)} MLflow run(s)")
        print(f"Most recent run: {os.path.dirname(run_paths[0])}")
        
        # Check metrics
        recent_run_dir = os.path.dirname(run_paths[0])
        metrics_dir = os.path.join(recent_run_dir, 'metrics')
        if os.path.exists(metrics_dir):
            metrics_files = os.listdir(metrics_dir)
            print(f"  Logged metrics: {', '.join(metrics_files)}")
    else:
        print("\nNo MLflow runs found")
else:
    print("\nMLflow directory not found")

print("\n" + "="*60)
print("DIAGNOSIS")
print("="*60)

# Read when my fixes were made
train_py_stat = os.stat('src/pipelines/train.py')
train_py_mod = datetime.fromtimestamp(train_py_stat.st_mtime)

model_stat = os.stat(model_path)
model_mod = datetime.fromtimestamp(model_stat.st_mtime)

print(f"\ntraining script (train.py) last modified: {train_py_mod}")
print(f"Model file last modified: {model_mod}")

if model_mod < train_py_mod:
    print("\n❌ MODEL IS OLDER THAN THE FIXED TRAINING SCRIPT!")
    print("   This means you trained BEFORE the fixes were applied.")
    print("   You need to retrain with the updated script.")
elif (model_mod - train_py_mod).total_seconds() < 60:
    print("\n⚠ Model was created around the same time as the script update")
    print("  Hard to tell if it used the fixes or not")
else:
    print("\n✓ Model was created AFTER the training script was fixed")
    print("  So it should have used the proper train/val split")
    print("  But performance is still poor, suggesting another issue...")

print("\n" + "="*60)
print("RECOMMENDATION")
print("="*60)

print("\nBased on the model's behavior (always predicting class 1),")
print("the most likely issues are:")
print()
print("1. Model was trained BEFORE my fixes were applied")
print("   → Solution: Retrain now with: python3 -m src.pipelines.train \\")
print("              --model googlenet --epochs 30 --lr 1e-4")
print()
print("2. Training used wrong data or had errors")
print("   → Solution: Check training output for errors, then retrain")
print()
print("3. There's a bug in the training loop or data loading")
print("   → Solution: I need to see your training output to diagnose")
print()
print("Please share:")
print("  - The full output from your training run")
print("  - Or retrain now and share the output")
