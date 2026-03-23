#!/bin/bash
#SBATCH --job-name=ghap2021
#SBATCH --output=/scratch/project_462001140/ammar/eccv/aria/scripts/ghap2021_%j.out
#SBATCH --account=project_462001140
#SBATCH --partition=standard
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

module load LUMI/25.03 partition/C

source /scratch/project_462000640/ammar/aq_net2/venv_pytorch_rocm/bin/activate

echo "=== Rebuild GHAP 2021 ==="
echo "Start: $(date)"
python3 -u /scratch/project_462001140/ammar/eccv/aria/scripts/rebuild_ghap_year.py 2021 10799203
echo "End: $(date)"
