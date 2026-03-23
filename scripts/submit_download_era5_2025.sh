#!/bin/bash
#SBATCH --job-name=era5_2025
#SBATCH --account=project_462001140
#SBATCH --partition=small
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/scratch/project_462001140/ammar/eccv/aria/scripts/era5_2025_%j.out

module purge
module load LUMI/25.03 partition/C
source /scratch/project_462000640/ammar/aq_net2/venv_pytorch_rocm/bin/activate

echo "=== ERA5 2025 Download (Jan-Mar, zero-shot inference) ==="
date
python3 /scratch/project_462001140/ammar/eccv/aria/scripts/download_era5_2025.py
echo "Done: $(date)"
