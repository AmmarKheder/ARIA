#!/bin/bash
#SBATCH --job-name=road_density
#SBATCH --output=/scratch/project_462001140/ammar/eccv/aria/scripts/road_density_%j.out
#SBATCH --account=project_462001140
#SBATCH --partition=standard
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

module load LUMI/25.03 partition/C
source /scratch/project_462000640/ammar/aq_net2/venv_pytorch_rocm/bin/activate

# Install osmium if needed
pip install osmium rasterio --quiet 2>/dev/null

echo "=== Build Road Density Proxy ==="
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "Work dir: /scratch/project_462001140/ammar/eccv/data/raw/osm_roads"

python3 -u /scratch/project_462001140/ammar/eccv/aria/scripts/build_road_density.py

echo "End: $(date)"
