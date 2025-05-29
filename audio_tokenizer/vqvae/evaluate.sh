#!/bin/bash
#SBATCH --job-name=snoupy_ddp
#SBATCH --time=20:00:00
#SBATCH --account=com-304
#SBATCH --qos=com-304
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=ddp_job.out
#SBATCH --error=ddp_job.err

source /work/com-304/new_environment/anaconda3/etc/profile.d/conda.sh
conda activate nanofm

# Use torchrun to launch DDP across GPUs
PYTHONPATH=. torchrun --nproc_per_node=2 audio_tokenizer/vqvae/evaluate.py
