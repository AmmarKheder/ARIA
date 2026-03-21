#!/bin/bash
# Submit all global data downloads
# Run from: /scratch/project_462001140/ammar/eccv/aria/scripts/

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Submitting all global data downloads ==="
echo ""

# 1. ERA5 global (6 jobs, one per year)
ERA5_JOB=$(sbatch --parsable submit_download_era5_global.sh)
echo "ERA5 global: job array $ERA5_JOB (2017-2022)"

# 2. CAMS EAC4 global (6 jobs)
CAMS_JOB=$(sbatch --parsable submit_download_cams_global.sh)
echo "CAMS global: job array $CAMS_JOB (2017-2022)"

# 3. GHAP global (6 jobs — these are the biggest!)
GHAP_JOB=$(sbatch --parsable submit_download_ghap_global.sh)
echo "GHAP global: job array $GHAP_JOB (2017-2022)"

# 4. Elevation (single job)
ELEV_JOB=$(sbatch --parsable submit_download_elevation_global.sh)
echo "Elevation global: job $ELEV_JOB"

echo ""
echo "Total: 19 jobs submitted"
echo "  ERA5: 6 years × ~45 GB/year = ~270 GB"
echo "  CAMS: 6 years × ~5 GB/year  = ~30 GB"
echo "  GHAP: 6 years × ~150 GB/year = ~900 GB (!!)"
echo "  Elev: ~3 GB download → ~10 GB zarr"
echo ""
echo "TOTAL ESTIMATED: ~1.2 TB"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Check quota: lfs quota -h /scratch/project_462001140"
