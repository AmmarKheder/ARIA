#!/bin/bash
#SBATCH --job-name=openaq_2023_24
#SBATCH --output=/scratch/project_462001140/ammar/eccv/aria/scripts/openaq_2023_24_%j.out
#SBATCH --account=project_462001140
#SBATCH --partition=standard
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

module load LUMI/25.03 partition/C
source /scratch/project_462000640/ammar/aq_net2/venv_pytorch_rocm/bin/activate

echo "=== OpenAQ 2023 + 2024 — Full resolution (same as 2025) ==="
echo "Start: $(date)"

echo ""
echo "--- OpenAQ 2023 ---"
python3 -u /scratch/project_462001140/ammar/eccv/aria/scripts/download_openaq_v2.py 2023

echo ""
echo "--- OpenAQ 2024 ---"
python3 -u /scratch/project_462001140/ammar/eccv/aria/scripts/download_openaq_v2.py 2024

echo ""
echo "=== DONE: $(date) ==="
