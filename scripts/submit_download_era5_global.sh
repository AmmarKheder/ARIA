#!/bin/bash
#SBATCH --job-name=dl_era5_global
#SBATCH --account=project_462001140
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=dl_era5_global_%j.out
#SBATCH --array=0-5

module load LUMI/25.03 partition/C

source /scratch/project_462000640/ammar/aq_net2/venv_pytorch_rocm/bin/activate

YEAR=$((2017 + SLURM_ARRAY_TASK_ID))

echo "=== Downloading ERA5 GLOBAL for year $YEAR ==="
echo "Start: $(date)"

python /scratch/project_462001140/ammar/eccv/aria/scripts/download_era5_global.py $YEAR

echo "End: $(date)"
