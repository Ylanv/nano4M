#!/bin/bash
#SBATCH --job-name=wavenet-360
#SBATCH --time=20:00:00
#SBATCH --account=com-304
#SBATCH --qos=com-304
#SBATCH --gres=gpu:2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=multi_node_job.out
#SBATCH --error=multi_node_job.err

# === Accept arguments ===
WANDB_KEY=$1        


# === Initialization ===
set -x
cat $0
export MASTER_PORT=25678
export MASTER_ADDR=$(hostname)
export WANDB_API_KEY=$WANDB_KEY
export NCCL_DEBUG=INFO

# === Conda === 
source /work/com-304/new_environment/anaconda3/etc/profile.d/conda.sh
conda init
conda activate nanofm

# === Run main script ===
srun bash -c "
  TORCHRUN_ARGS=\"--node-rank=\${SLURM_PROCID} \
     --master-addr=\${MASTER_ADDR} \
     --master-port=\${MASTER_PORT} \
     --nnodes=\${SLURM_NNODES} \
     --nproc-per-node=2\"

  echo \${SLURM_PROCID}
  echo \${TORCHRUN_ARGS}
  echo \${SLURMD_NODENAME}

  PYTHONPATH=. torchrun \${TORCHRUN_ARGS} audio_tokenizer/wavenet/train_wavenet.py
"