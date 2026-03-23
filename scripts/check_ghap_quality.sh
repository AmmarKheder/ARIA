#!/bin/bash
#SBATCH --job-name=check_ghap
#SBATCH --output=/scratch/project_462001140/ammar/eccv/aria/scripts/check_ghap_%j.out
#SBATCH --account=project_462001140
#SBATCH --partition=standard
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G

module load LUMI/25.03 partition/C
source /scratch/project_462000640/ammar/aq_net2/venv_pytorch_rocm/bin/activate

python3 -c "
import zarr, numpy as np, os
from concurrent.futures import ThreadPoolExecutor

base = '/scratch/project_462001140/ammar/eccv/data/zarr/ghap_global_daily'

# Sample points: Beijing, India, Europe, USA, China-South, Africa
regions = {
    'Beijing':  (5010, 29639),
    'India':    (6500, 25000),
    'Europe':   (4100, 18200),
    'E.USA':    (5000, 10600),
    'S.China':  (6200, 29500),
    'W.Africa': (8000, 18000),
}
days = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 364]

print('=== GHAP Daily Zarr — Full Quality Report ===')
print()

for yr in [2018, 2019, 2020, 2021, 2022]:
    p = f'{base}/{yr}.zarr'
    if not os.path.exists(p):
        print(f'{yr}: MISSING ❌')
        continue

    z = zarr.open(p, mode='r', zarr_format=2)
    print(f'{yr}: shape={z.shape}, chunks={z.chunks}')

    all_ok = True
    for name, (r, c) in regions.items():
        vals = [float(z[d, r, c]) for d in days]
        nz = sum(1 for v in vals if v > 0)
        mx = max(vals)
        status = '✅' if nz >= 10 else ('⚠️' if nz >= 6 else '❌')
        if nz < 10: all_ok = False
        print(f'  {name:10s}: {nz}/{len(days)} days non-zero, max={mx:.1f}  {status}')

    # Check for NaN
    sample = z[100, 5000:5100, 25000:25100]
    nan_pct = np.isnan(sample).mean() * 100
    nan_status = '✅' if nan_pct == 0 else '❌'
    print(f'  NaN check:  {nan_pct:.1f}% NaN in sample  {nan_status}')
    print(f'  => Overall: {\"CLEAN ✅\" if all_ok and nan_pct==0 else \"ISSUES ⚠️\"}')
    print()
"
