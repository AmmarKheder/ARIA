#!/bin/bash
#SBATCH --job-name=aria_global
#SBATCH --output=/scratch/project_462001140/ammar/eccv/topoflow_global/scripts/aria_global_%j.out
#SBATCH --account=project_462001140
#SBATCH --partition=standard-g
#SBATCH --time=2-00:00:00
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G

module load LUMI/25.03 partition/G rocm/6.0.3

export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3
export MIOPEN_USER_DB_PATH=/tmp/miopen_cache_global_${SLURM_JOB_ID}
export MIOPEN_DISABLE_CACHE=0

# Create MIOpen cache dirs on all nodes
srun --ntasks-per-node=1 mkdir -p $MIOPEN_USER_DB_PATH

source /scratch/project_462000640/ammar/aq_net2/venv_pytorch_rocm/bin/activate

export CKPT_DIR=/scratch/project_462001140/ammar/eccv/topoflow_global/checkpoints_global
export LOG_DIR=/scratch/project_462001140/ammar/eccv/topoflow_global/logs_global
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=29500

echo "=== ARIA Global Training ==="
echo "Nodes: $SLURM_NNODES, GPUs/node: 8, Total GPUs: $((SLURM_NNODES * 8))"
echo "Start: $(date)"

srun python /scratch/project_462001140/ammar/eccv/topoflow_global/train_global.py \
    --config /scratch/project_462001140/ammar/eccv/topoflow_global/configs/global_pretrain.yaml

echo "End: $(date)"
