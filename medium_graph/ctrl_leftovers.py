"""
Prints the SLURM --array list for submit_lowlabel_ctrl_fine.sbatch (or, with
--coarse, for submit_lowlabel_ctrl.sbatch) covering every control task that
has not logged 'Task N done.' in any results/lowlabel_ctrl_*.log.

    sbatch --array=$(python ctrl_leftovers.py) submit_lowlabel_ctrl_fine.sbatch
"""
import argparse, glob, re

ap = argparse.ArgumentParser()
ap.add_argument('--coarse', action='store_true', help='ids for the 7-lambda-per-task array')
ap.add_argument('--prefix', default='results/lowlabel_ctrl_', help='log prefix (results/transfer_ for E4)')
ap.add_argument('--n', type=int, default=128, help='number of tasks in the coarse array')
a = ap.parse_args()

done = set()
for f in glob.glob(f'{a.prefix}*.log'):
    m = re.search(r'_(\d+)\.log$', f)
    if m and any(l.startswith('Task ') and l.strip().endswith('done.') for l in open(f, errors='ignore')):
        done.add(int(m.group(1)))
left = [i for i in range(a.n) if i not in done]
if a.coarse:
    print(','.join(map(str, left)))
else:
    print(','.join(f'{7*i}-{7*i+6}' for i in left))
