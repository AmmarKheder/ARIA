#!/bin/bash
#SBATCH --job-name=dl_ghap_global
#SBATCH --account=project_462001140
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=dl_ghap_global_%j.out
#SBATCH --array=0-5

module load LUMI/25.03 partition/C

source /scratch/project_462000640/ammar/aq_net2/venv_pytorch_rocm/bin/activate

YEAR=$((2017 + SLURM_ARRAY_TASK_ID))

echo "=== Downloading GHAP GLOBAL D1K for year $YEAR ==="
echo "Start: $(date)"
echo "WARNING: Each year ~100-200 GB compressed. Ensure sufficient quota."

python /scratch/project_462001140/ammar/eccv/aria/scripts/download_ghap_global.py $YEAR

echo "End: $(date)"
