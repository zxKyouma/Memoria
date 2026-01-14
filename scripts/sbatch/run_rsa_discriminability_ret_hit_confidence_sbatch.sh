#!/bin/bash
#SBATCH --job-name=rsa_hit_conf
#SBATCH --partition=comp
#SBATCH --output=/home/zlow/by84/Brandon/memoria/logs/rsa_hit_conf_%A.out
#SBATCH --error=/home/zlow/by84/Brandon/memoria/logs/rsa_hit_conf_%A.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

cd /home/zlow/by84/Brandon/memoria

module load miniforge3/24.3.0-0
conda activate nnunet

export MKL_THREADING_LAYER=SEQ
export OMP_NUM_THREADS=1

python code/fmri/analysis/compute_rsa_discriminability.py \
  --analysis ret_hit_confidence \
  --condition-a ret_hit_sure \
  --condition-b ret_hit_unsure \
  --min-trials 4 \
  --output-dir id_analysis_stats/discriminability_ret_hit_confidence \
  --voxel-subsample min
