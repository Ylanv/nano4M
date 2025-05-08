#!/bin/bash
#SBATCH --job-name=snoupy         # Change as needed
#SBATCH --time=20:00:00
#SBATCH --account=com-304
#SBATCH --qos=com-304
#SBATCH --gres=gpu:1                  # Request 2 GPUs
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4               # Adjust CPU allocation if needed
#SBATCH --output=interactive_job.out    # Output log file
#SBATCH --error=interactive_job.err     # Error log file

source /work/com-304/new_environment/anaconda3/etc/profile.d/conda.sh
conda init
conda activate nanofm 
PYTHONPATH=. python audio_tokenizer/vqvae/train_raw.py 
