#!/bin/bash
#SBATCH --job-name=dl_2025
#SBATCH --output=/scratch/project_462001140/ammar/eccv/aria/scripts/dl_2025_%j.out
#SBATCH --account=project_462001140
#SBATCH --partition=standard
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

module load LUMI/25.03 partition/C
source /scratch/project_462000640/ammar/aq_net2/venv_pytorch_rocm/bin/activate

echo "=== Download CAMS 2025 ==="
echo "Start: $(date)"
python3 -u /scratch/project_462001140/ammar/eccv/aria/scripts/download_cams_2025.py
echo "End: $(date)"
