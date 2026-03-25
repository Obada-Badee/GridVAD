import os
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from grid_vad.eval.eval_runner import main as run_eval

def run_server():
    """
    Wrapper to run evaluation on server with custom cache and model.
    """
    # 1. Env Setup
    cache_dir = "/ibex/user/hamidme/cache_dir/"
    os.environ["HF_HOME"] = cache_dir
    os.environ["TORCH_HOME"] = cache_dir
    
    print(f"Set HF/Torch cache to: {cache_dir}")
    
    # 2. Defaults for server run
    # You can change these
    model_id = "Qwen/Qwen3-VL-30B-A3B-Instruct" 
    # Note: User mentioned 'Qwen3', if that exists on HF, swap the string.
    # Qwen2.5 is the current latest standard.
    
    # Check arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="shanghaitech or ucsd_ped2")
    parser.add_argument("--dataset_root", required=True, help="Path to dataset directory")
    parser.add_argument("--model", default=model_id, help="HF Model ID")
    
    args, unknown = parser.parse_known_args()
    
    # Pass through to eval_runner
    # We construct the sys.argv that eval_runner expects
    sys.argv = [sys.argv[0]]
    sys.argv.extend(["--dataset", args.dataset])
    sys.argv.extend(["--dataset_root", args.dataset_root])
    sys.argv.extend(["--vlm_model", args.model])
    
    # Flags to tell eval_runner to use LocalVLMClient? 
    # Currently eval_runner uses VLMClient(client=None).
    # We need to update eval_runner to handle "if vlm_model looks like HF path, use LocalVLMClient"
    
    print("Launching evaluation...")
    from grid_vad.eval import eval_runner
    
    # Monkey patch or modify eval_runner to use local client if needed
    # Or better, just fix eval_runner.py to detect usage.
    # For now, let's assume we modify eval_runner.py next.
    
    eval_runner.main()

if __name__ == "__main__":
    run_server()
