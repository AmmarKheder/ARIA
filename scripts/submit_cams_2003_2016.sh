#!/bin/bash
#SBATCH --job-name=dl_cams_03
#SBATCH --account=project_462001140
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/project_462001140/ammar/eccv/aria/scripts/dl_cams_03_%A_%a.out
#SBATCH --array=0-13

module load LUMI/25.03 partition/C

source /scratch/project_462000640/ammar/aq_net2/venv_pytorch_rocm/bin/activate

YEAR=$((2003 + SLURM_ARRAY_TASK_ID))

OUT=/scratch/project_462001140/ammar/eccv/data/zarr/cams_global_daily/${YEAR}.zarr
if [ -d "$OUT" ]; then
    echo "Already exists: $OUT — skipping"
    exit 0
fi

echo "=== Downloading CAMS EAC4 GLOBAL for year $YEAR ==="
echo "Start: $(date)"

python /scratch/project_462001140/ammar/eccv/aria/scripts/download_cams_global.py $YEAR

echo "End: $(date)"
