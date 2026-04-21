import os
import sys
import subprocess

def run_sweep(base_cmd):
    print("==========================================================")
    print("Starting Delayed Lambda Sweep (Start at 500) for:")
    print(base_cmd)
    print("==========================================================")
    
    # Run Baseline
    print("Running Baseline (REG: False)...")
    subprocess.run(base_cmd, shell=True)
    
    # Run Sweeps
    for val in ["0.01", "0.05", "0.1", "0.5", "1.0"]:
        print(f"\nRunning with Delayed Regularization (lambda={val}, start=500)...")
        # Added --reg_start_epoch 500 
        # If you meant computing it just once at epoch 500 and freezing it, we use a huge update_freq
        # If you meant to compute it at 500 and continue dynamically updating, remove --reg_update_freq
        delayed_cmd = f"{base_cmd} --use_reg --lambda_val {val} --reg_start_epoch 500 --reg_update_freq 5000"
        print(f"Command: {delayed_cmd}")
        subprocess.run(delayed_cmd, shell=True)
        
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please pass the GPU device ID. Example: python run_gnn_sweep_delayed.py 0")
        sys.exit(1)
        
    device = sys.argv[1]
    sh_file = "run_gnn_sweep.sh"
    
    if not os.path.exists(sh_file):
        print(f"Could not find {sh_file} in the current directory.")
        sys.exit(1)
        
    with open(sh_file, "r") as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if line.startswith("./sweep_lambda.sh"):
            # Extract the command inside the quotes
            start_idx = line.find('"')
            end_idx = line.rfind('"')
            if start_idx != -1 and end_idx != -1:
                cmd = line[start_idx+1:end_idx]
                cmd = cmd.replace("$1", device)
                run_sweep(cmd)
