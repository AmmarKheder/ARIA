#!/bin/bash
#SBATCH --job-name=aria_debug
#SBATCH --output=/scratch/project_462001140/ammar/eccv/aria/scripts/aria_debug_%j.out
#SBATCH --account=project_462001140
#SBATCH --partition=standard-g
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G

module load LUMI/25.03 partition/G rocm/6.0.3

export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export MIOPEN_USER_DB_PATH=/tmp/miopen_cache_debug_${SLURM_JOB_ID}
export PYTHONUNBUFFERED=1

srun --ntasks-per-node=1 mkdir -p $MIOPEN_USER_DB_PATH
source /scratch/project_462000640/ammar/aq_net2/venv_pytorch_rocm/bin/activate

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=29500

echo "=== ARIA Debug (1 node, 8 GPUs) ==="
echo "Start: $(date)"

srun python /scratch/project_462001140/ammar/eccv/aria/train_global.py \
    --config /scratch/project_462001140/ammar/eccv/aria/configs/global_pretrain.yaml

echo "End: $(date)"
