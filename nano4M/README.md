# COM-304 - nano4M Project  

Welcome to the nano4M project exercises!  

In this project, you will conduct hands-on experiments to implement and train nano4M, a multi-modal foundation model capable of any-to-any generation.  

This project consists of three parts:  

1) In this first part, we start by implementing the necessary building blocks to construct an autoregressive Transformer, like GPT.
2) In part 2, we will build a masked model in the style of MaskGIT.  
3) In part 3, we will build a simple 4M-like multimodal model.

Course materials for each part will be released in stages, see the schedule below.  

### Instructions

The instructions for each of these three parts are provided in the notebooks, which you can find under `./notebooks/`. They will introduce the problem statement to you, explain what parts in the codebase need to be completed, and you will use them to perform inference on the trained models. You will be asked to run the cells in those notebooks, provide answers to questions, etc. Each notebook will count towards 10% of your nano4M project grade, meaning that the remaining 70% will come from your extensions. The notebooks are to be completed and submitted as a group.

## **Important Dates**  

Below is the completion and homework submission timeline for each part. Please refer to Moodle for further updates and instructions.  

### **Part 1**  
- **Release:** Tue 25.3.2025  
- **Due:** By 23:59 on Fri 4.4.2025  

### **Part 2**  
- **Release:** Tue 1.4.2025  
- **Due:** By 23:59 on Fri 11.4.2025  

### **Part 3**  
- **Release:** Tue 8.4.2025  
- **Due:** By 23:59 on Fri 18.4.2025  

### **Progress Report**  
- **Due:** By 23:59 on Fri 18.4.2025  

## **Installation**  

To begin the experiments (and submit jobs), we first need to install the required packages and dependencies. For ease of installation and running experiments directly, we provide a pre-installed Anaconda environment that you can activate as follows:  

```bash
source /work/com-304/new_environment/anaconda3/etc/profile.d/conda.sh
conda init
conda activate nanofm
```

Alternatively, you can install the environment yourself by running [setup_env.sh](setup_env.sh)
```bash
bash setup_env.sh
```

## Getting Started

In Part 1, we will implement the building blocks of autoregressive models and train them on language and image modeling tasks.

You will primarily run the following files:
1. Jupyter notebook: `nano4M/notebooks/COM304_FM_part1_nanoGPT.ipynb` 
   - Usage: Introduction of **Part 1** and Inference (post-training result generation and analysis).
2. Main training script: `run_training.py` 
   - Usage: Train your models after implementing the building blocks (refer to the notebook for more details).

### Jupyter notebook `notebooks/COM304_FM_part1_nanoGPT.ipynb`:
To use the Jupyter notebook, activate the `nano4m` environment and run the notebook. Follow the same steps outlined in [4M_Tutorial Environment Setup](4M_Tutorial/Environment.md) to launch the notebook in a browser.

### Main training script `run_training.py`:

You can run the training job in one of two ways:

1. **Interactively using `srun`** – great for debugging.
2. **Using a SLURM batch script** – better for running longer jobs.

> **Before you begin**:  
> Make sure to have your Weights & Biases (W&B) account and obtain your W&B API key.  
> Follow the instructions in **Section 1.3 (Weights & Biases setup)** of the Jupyter Notebook.

---

#### Option 1: Run Interactively via `srun`

Start an interactive session on a compute node (eg, 2 GPUs case):

```bash
srun -t 120 -A com-304 --qos=com-304 --gres=gpu:2 --mem=16G --pty bash
```
Then, on the compute node:

```bash
conda activate nanofm
wandb login
OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 run_training.py --config cfgs/nanoGPT/tinystories_d8w512.yaml
```
> **Note:**  
> To run the job on **one GPU**, make sure to:
> - Adjust the `--gres=gpu:1` option in the `srun` command, and  
> - Set `--nproc_per_node=1` in the `torchrun` command.

#### Option 2: Submit as a Batch Job via SLURM
You can use the provided submit_job.sh script to request GPUs and launch training.

Run:
```bash
sbatch submit_job.sh <config_file> <your_wandb_key> <num_gpus>
```
Replace the placeholders as follows:

- <config_file> — Path to your YAML config file

- <your_wandb_key> — Your W&B API key

- <num_gpus> — Set to 1 or 2 depending on your setup

Example Usage:
```bash
sbatch submit_job.sh cfgs/nanoGPT/tinystories_d8w512.yaml abcdef1234567890 2
```



