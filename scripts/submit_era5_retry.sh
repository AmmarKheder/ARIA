#!/bin/bash
#SBATCH --job-name=dl_era5_retry
#SBATCH --output=/scratch/project_462001140/ammar/eccv/aria/scripts/dl_era5_retry_%A_%a.out
#SBATCH --account=project_462001140
#SBATCH --partition=small
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --array=0-3

YEARS=(2004 2010 2014 2016)
YEAR=${YEARS[$SLURM_ARRAY_TASK_ID]}

echo "=== ERA5 GLOBAL retry $YEAR ==="
echo "Start: $(date)"

module load LUMI/25.03 partition/C
source /scratch/project_462000640/ammar/aq_net2/venv_pytorch_rocm/bin/activate

rm -rf /scratch/project_462001140/ammar/eccv/data/zarr/era5_global_daily/${YEAR}.zarr

python /scratch/project_462001140/ammar/eccv/aria/scripts/download_era5_global.py $YEAR

echo "End: $(date)"
