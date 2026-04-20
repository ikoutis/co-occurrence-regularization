with open("run_gnn.sh", "r") as f:
    lines = f.readlines()

with open("run_gnn_sweep.sh", "w", newline='\n') as f:
    for line in lines:
        if line.startswith("python main.py"):
            f.write(f'./sweep_lambda.sh "{line.strip()}"\n')
        else:
            f.write(line)
