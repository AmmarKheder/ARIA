#!/bin/bash
#SBATCH --job-name=aria_frames
#SBATCH --output=/scratch/project_462001140/ammar/eccv/aria/scripts/aria_frames_%j.out
#SBATCH --account=project_462001140
#SBATCH --partition=small
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

module load LUMI/25.03

source /scratch/project_462000640/ammar/aq_net2/venv_pytorch_rocm/bin/activate

FRAMES_DIR=/scratch/project_462001140/ammar/eccv/aria/video_frames

echo "=== ARIA Frame Generator ==="
echo "Start: $(date)"

# Global overview
python /scratch/project_462001140/ammar/eccv/aria/scripts/generate_video_frames.py \
    --input  /scratch/project_462000640/ammar/zeroshot_2025_comparison.npz \
    --outdir ${FRAMES_DIR}/global \
    --region global \
    --fps 12

# China zoom (high-interest region)
python /scratch/project_462001140/ammar/eccv/aria/scripts/generate_video_frames.py \
    --input  /scratch/project_462000640/ammar/zeroshot_2025_comparison.npz \
    --outdir ${FRAMES_DIR}/china \
    --region china \
    --fps 12

# India zoom
python /scratch/project_462001140/ammar/eccv/aria/scripts/generate_video_frames.py \
    --input  /scratch/project_462000640/ammar/zeroshot_2025_comparison.npz \
    --outdir ${FRAMES_DIR}/india \
    --region india \
    --fps 12

# Europe zoom
python /scratch/project_462001140/ammar/eccv/aria/scripts/generate_video_frames.py \
    --input  /scratch/project_462000640/ammar/zeroshot_2025_comparison.npz \
    --outdir ${FRAMES_DIR}/europe \
    --region europe \
    --fps 12

echo "End: $(date)"
echo ""
echo "=== ffmpeg assembly commands ==="
for region in global china india europe; do
    echo "ffmpeg -framerate 12 -pattern_type glob -i '${FRAMES_DIR}/${region}/frame_*.png' \\"
    echo "       -vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' \\"
    echo "       -vcodec libx264 -pix_fmt yuv420p -crf 18 \\"
    echo "       /scratch/project_462001140/ammar/eccv/aria/aria_vs_cams_${region}.mp4"
    echo ""
done
