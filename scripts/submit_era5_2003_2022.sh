#!/bin/bash
#SBATCH --job-name=dl_era5_gl
#SBATCH --account=project_462001140
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/project_462001140/ammar/eccv/aria/scripts/dl_era5_gl_%A_%a.out
#SBATCH --array=0-19

module load LUMI/25.03 partition/C

source /scratch/project_462000640/ammar/aq_net2/venv_pytorch_rocm/bin/activate

YEAR=$((2003 + SLURM_ARRAY_TASK_ID))

OUT=/scratch/project_462001140/ammar/eccv/data/zarr/era5_global_daily/${YEAR}.zarr
# Skip check is in Python (validates chunk files exist, not just dir)

echo "=== Downloading ERA5 GLOBAL for year $YEAR ==="
echo "Start: $(date)"

python /scratch/project_462001140/ammar/eccv/aria/scripts/download_era5_global.py $YEAR

echo "End: $(date)"
