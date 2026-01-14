#!/bin/bash
#SBATCH --job-name=rsa_task_bvae
#SBATCH --partition=comp
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/zlow/by84/Brandon/memoria/logs/rsa_task_bvae_%j.out
#SBATCH --error=/home/zlow/by84/Brandon/memoria/logs/rsa_task_bvae_%j.err

cd /home/zlow/by84/Brandon/memoria

module load miniforge3/24.3.0-0
conda activate nnunet

python derivatives/bvae_project/compute_rsa_checkpoint.py \
  --bids-dir /home/zlow/by84/Brandon/memoria \
  --session-label 2 \
  --task-label ret \
  --model-path /home/zlow/by84/Brandon/memoria/derivatives/bvae_project/outputs_task/bvae_task_beta4.0_latent32.pth \
  --image-dir /home/zlow/by84/Brandon/memoria/derivatives/bvae_project/data/all_images \
  --output-dir /home/zlow/by84/Brandon/memoria/rsa_results_task_bvae
