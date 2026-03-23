#!/bin/bash
#SBATCH --job-name=stations_2023_24
#SBATCH --output=/scratch/project_462001140/ammar/eccv/aria/scripts/stations_2023_24_%j.out
#SBATCH --account=project_462001140
#SBATCH --partition=standard
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

module load LUMI/25.03 partition/C
source /scratch/project_462000640/ammar/aq_net2/venv_pytorch_rocm/bin/activate

SCRIPTS=/scratch/project_462001140/ammar/eccv/aria/scripts

echo "============================================"
echo "Stations 2023-2024 — Unified format (same as 2025)"
echo "Start: $(date)"
echo "============================================"

echo ""
echo "=== OpenAQ 2023 (S3, full resolution) ==="
python3 -u $SCRIPTS/download_openaq_v2.py 2023

echo ""
echo "=== OpenAQ 2024 (S3, full resolution) ==="
python3 -u $SCRIPTS/download_openaq_v2.py 2024

echo ""
echo "=== CNEMC 2023 (quotsoft, daily) ==="
python3 -u $SCRIPTS/download_cnemc_v2.py 2023

echo ""
echo "=== CNEMC 2024 (quotsoft, daily) ==="
python3 -u $SCRIPTS/download_cnemc_v2.py 2024

echo ""
echo "============================================"
echo "DONE: $(date)"
echo "============================================"
