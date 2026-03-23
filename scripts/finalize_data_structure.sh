#!/bin/bash
# Run after download jobs complete to finalize aria/data/ structure

STATIONS_DIR="/scratch/project_462001140/ammar/eccv/aria/data/stations"

echo "=== Finalizing aria/data/ structure ==="

# OpenAQ station data
for yr in 2023 2024 2025; do
    f="/scratch/project_462000640/ammar/openaq_${yr}_global_pm25.npz"
    if [ -f "$f" ]; then
        ln -sfn "$f" "${STATIONS_DIR}/openaq_${yr}_pm25.npz"
        echo "  Linked: openaq_${yr}_pm25.npz"
    fi
done

# CNEMC station data
for yr in 2023 2024 2025; do
    f="/scratch/project_462000640/ammar/cnemc_${yr}_pm25.npz"
    if [ -f "$f" ]; then
        ln -sfn "$f" "${STATIONS_DIR}/cnemc_${yr}_pm25.npz"
        echo "  Linked: cnemc_${yr}_pm25.npz"
    fi
done

echo ""
echo "=== aria/data/ structure ==="
ls -la /scratch/project_462001140/ammar/eccv/aria/data/
echo ""
echo "=== Stations ==="
ls -la ${STATIONS_DIR}/

echo ""
echo "=== ERA5 years ==="
ls /scratch/project_462001140/ammar/eccv/aria/data/era5/ | sort

echo ""
echo "=== CAMS years ==="
ls /scratch/project_462001140/ammar/eccv/aria/data/cams/ | sort

echo ""
echo "=== GHAP years ==="
ls /scratch/project_462001140/ammar/eccv/aria/data/ghap/ | sort
