import os
import sys
import subprocess

def run_sweep(base_cmd):
    print("==========================================================")
    print("Starting Lambda Sweep for Baseline Command:")
    print(base_cmd)
    print("==========================================================")
    
    # Run Baseline
    print("Running Baseline (REG: False)...")
    subprocess.run(base_cmd, shell=True)
    
    # Run Sweeps
    for val in ["0.01", "0.05", "0.1", "0.5", "1.0"]:
        print(f"\nRunning with Dynamic Regularization (lambda={val})...")
        subprocess.run(f"{base_cmd} --use_reg --lambda_val {val}", shell=True)
        
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please pass the GPU device ID. Example: python run_gnn_sweep_windows.py 0")
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
            # E.g. ./sweep_lambda.sh "python main.py ... --device $1 --ln"
            start_idx = line.find('"')
            end_idx = line.rfind('"')
            if start_idx != -1 and end_idx != -1:
                cmd = line[start_idx+1:end_idx]
                # Replace the bash variable $1 with our device ID
                cmd = cmd.replace("$1", device)
                run_sweep(cmd)
