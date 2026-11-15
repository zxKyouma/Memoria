#!/bin/bash
#SBATCH -J nibetaseries-array
#SBATCH --time=12:00:00
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -o ${MEMORIA_ROOT}/logs/nibetaseries/%x-%A-%a.out
#SBATCH -e ${MEMORIA_ROOT}/logs/nibetaseries/%x-%A-%a.err

# --- SETUP: Define paths and variables ---
PROJECT_DIR="${MEMORIA_ROOT}"
SCRATCH_DIR="${SCRATCH_DIR:-/tmp}"

# --- INPUT PATHS ---
BIDS_DIR="${PROJECT_DIR}"
FMRIPREP_DIR="${PROJECT_DIR}/derivatives/fMRIprepNoSbref"

# --- OUTPUT PATH ---
NIBETASERIES_DIR="${PROJECT_DIR}/derivatives/nibetaseries"
mkdir -p ${NIBETASERIES_DIR}
mkdir -p ${MEMORIA_ROOT}/logs/nibetaseries

NIBETASERIES_IMG="${PROJECT_DIR}/singularity_images/nibetaseries-v0.6.0.simg"

SINGULARITY_WORK_DIR="${SCRATCH_DIR}/nibetaseries_work/${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p ${SINGULARITY_WORK_DIR}

# --- JOB ARRAY LOGIC ---
subject=$( sed -n -E "$((${SLURM_ARRAY_TASK_ID} + 1))s/sub-(\S*)\>.*/\1/gp" ${BIDS_DIR}/participants.tsv )

# --- THE NIBETASERIES COMMAND-LINE ARGUMENTS ---
# NOTE: No special database flags are needed here.
# The nibs command will automatically find the pre-generated database
# in /data/.pybids/ because it's the default location.
cmd="singularity run --cleanenv \
  -B ${BIDS_DIR}:/data \
  -B ${FMRIPREP_DIR}:/fmriprep \
  -B ${NIBETASERIES_DIR}:/out \
  -B ${SINGULARITY_WORK_DIR}:/work \
  ${NIBETASERIES_IMG} \
  nibs \
    -w /work \
    -t forgetexp \
    -sp T1w \
    -c csf white_matter trans_x trans_y trans_z rot_x rot_y rot_z a_comp_cor_00 a_comp_cor_01 a_comp_cor_02 a_comp_cor_03 a_comp_cor_04 a_comp_cor_05 \
    --participant-label $subject \
    --nthreads 8 \
    --hrf-model 'glover' \
    /data /fmriprep /out participant"

# --- JOB EXECUTION ---
echo "--- JOB START ---"
echo "Running NiBetaSeries task ${SLURM_ARRAY_TASK_ID} for sub-${subject}"
echo "Commandline: $cmd"
echo "---"

eval $cmd
exitcode=$?

echo "--- JOB END ---"
echo "sub-$subject   ${SLURM_ARRAY_TASK_ID}    $exitcode" \
      >> ${SLURM_JOB_NAME}.${SLURM_ARRAY_JOB_ID}.tsv
echo "Finished task ${SLURM_ARRAY_TASK_ID} for sub-${subject} with exit code $exitcode"

rm -rf $SINGULARITY_WORK_DIR
echo "Cleaned up work directory: ${SINGULARITY_WORK_DIR}"

exit $exitcode
