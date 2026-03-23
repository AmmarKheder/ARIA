#!/bin/bash
#SBATCH --job-name=aria_emission_proxies
#SBATCH --account=project_462001140
#SBATCH --partition=small
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/project_462001140/ammar/eccv/aria/scripts/emission_proxies_%j.out

module purge
module load LUMI/25.03 partition/C
source /scratch/project_462000640/ammar/aq_net2/venv_pytorch_rocm/bin/activate

echo "=== Download & Process Emission Proxies ==="
echo "Start: $(date)"
echo "Node: $(hostname)"

mkdir -p /scratch/project_462000640/ammar/tmp/emission_proxies

python3 /scratch/project_462001140/ammar/eccv/aria/scripts/download_emission_proxies.py

echo ""
echo "Done: $(date)"
