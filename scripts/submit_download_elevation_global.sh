#!/bin/bash
#SBATCH --job-name=dl_elev_global
#SBATCH --account=project_462001140
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=dl_elev_global_%j.out

module load LUMI/25.03 partition/C

source /scratch/project_462000640/ammar/aq_net2/venv_pytorch_rocm/bin/activate

echo "=== Downloading GMTED2010 GLOBAL elevation ==="
echo "Start: $(date)"

python /scratch/project_462001140/ammar/eccv/aria/scripts/download_elevation_global.py

echo "End: $(date)"
