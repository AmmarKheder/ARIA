#!/bin/bash
#SBATCH --job-name=dl_proxies
#SBATCH --account=project_462001140
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project_462000640/ammar/dl_emission_proxies_%j.out

module load LUMI/25.03 partition/C
source /scratch/project_462000640/ammar/aq_net2/venv_pytorch_rocm/bin/activate

# Install rasterio if needed
pip install rasterio --quiet

echo "Start: $(date)"
python /scratch/project_462001140/ammar/eccv/aria/scripts/download_emission_proxies.py
echo "End: $(date)"
