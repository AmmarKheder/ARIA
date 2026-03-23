#!/bin/bash
#SBATCH --job-name=dl_era5_cams
#SBATCH --output=/scratch/project_462001140/ammar/eccv/aria/scripts/dl_era5_cams_%j.out
#SBATCH --account=project_462001140
#SBATCH --partition=standard
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

module load LUMI/25.03 partition/C
source /scratch/project_462000640/ammar/aq_net2/venv_pytorch_rocm/bin/activate

echo "=== Download ERA5 + CAMS 2023 and 2024 ==="
echo "Start: $(date)"

# ERA5 2023
echo ""
echo "--- ERA5 2023 ---"
python3 -u /scratch/project_462001140/ammar/eccv/aria/scripts/download_era5_global.py 2023

# ERA5 2024
echo ""
echo "--- ERA5 2024 ---"
python3 -u /scratch/project_462001140/ammar/eccv/aria/scripts/download_era5_global.py 2024

# CAMS 2023
echo ""
echo "--- CAMS 2023 ---"
python3 -u /scratch/project_462001140/ammar/eccv/aria/scripts/download_cams_global.py 2023

# CAMS 2024
echo ""
echo "--- CAMS 2024 ---"
python3 -u /scratch/project_462001140/ammar/eccv/aria/scripts/download_cams_global.py 2024

echo ""
echo "=== DONE: $(date) ==="
