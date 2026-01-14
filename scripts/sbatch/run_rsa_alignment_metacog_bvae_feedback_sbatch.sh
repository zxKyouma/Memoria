#!/bin/bash
#SBATCH --job-name=rsa_metacog_bvae_fb
#SBATCH --partition=comp
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/zlow/by84/Brandon/memoria/logs/rsa_metacog_bvae_fb_%j.out
#SBATCH --error=/home/zlow/by84/Brandon/memoria/logs/rsa_metacog_bvae_fb_%j.err

cd /home/zlow/by84/Brandon/memoria

module load miniforge3/24.3.0-0
conda activate nnunet

python derivatives/bvae_project/compute_rsa_checkpoint.py \
  --bids-dir /home/zlow/by84/Brandon/memoria \
  --session-label 2 \
  --task-label retRatingFeedback \
  --model-path /home/zlow/by84/Brandon/memoria/derivatives/bvae_project/outputs_metacog/bvae_metacog_beta4.0_latent32.pth \
  --image-dir /home/zlow/by84/Brandon/memoria/derivatives/bvae_project/data/all_images \
  --context-label Feedback \
  --output-dir /home/zlow/by84/Brandon/memoria/rsa_results_metacog_bvae_feedback
