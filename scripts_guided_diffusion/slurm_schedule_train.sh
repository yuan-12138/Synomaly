#!/bin/sh

#SBATCH --job-name=lits13  # Job name
#SBATCH --output=slurm_output/slurm-%A.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=slurm_output/slurm-%A.err  # Standard error of the script
#SBATCH --time=7-00:00:00  # Limit on the total run time (format: days-hours:minutes:seconds)
#SBATCH --gres=gpu:1  # Number of GPUs if needed
#SBATCH --cpus-per-task=4  # Number of CPUs (Don't use more than 24 per GPU)
#SBATCH --mem=126G  # Memory in GB (Don't use more than 126G per GPU)

# ml avail  # list available modules
ml cuda  # load CUDA module
# ml miniconda3  # load miniconda module

conda --version  # output conda version
nvcc --version  # output cuda version

## activate corresponding environment
#source /home/guests/lucie_huang/.bashrc
#conda deactivate # If you launch your script from a terminal where your environment is already loaded, conda won't activate the environment. This guards against that. Not necessary if you always run this script from a clean terminal
#conda activate /home/guests/lucie_huang/miniconda3/envs/diffusion_env2
conda info --envs

python scripts_guided_diffusion/diffusion_training.py args_lits_13

ml -cuda # unload CUDA module
# ml -miniconda3  # unload miniconda module