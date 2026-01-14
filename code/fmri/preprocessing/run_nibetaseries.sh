#!/bin/bash
#SBATCH -J nibetaseries-nosm-array             # ADAPTED: Job name for this analysis
#SBATCH --time=96:00:00                       # ADAPTED: NiBetaSeries is much faster (e.g., 2 hours per subject)
#SBATCH -n 1
#SBATCH --cpus-per-task=8                     # ADAPTED: 8 CPUs is plenty for nibetaseries
#SBATCH --mem=128G                             # ADAPTED: 24 GB of total memory is a safe amount
# Outputs ----------------------------------
# ADAPTED: Log files will be saved in a new 'nibetaseries_nosm' subfolder in your logs
#SBATCH -o /fs04/by84/Brandon/memoria/logs/nibetaseries_nosm/%x-%A-%a.out
#SBATCH -e /fs04/by84/Brandon/memoria/logs/nibetaseries_nosm/%x-%A-%a.err
#SBATCH --mail-user=zlow0014@student.monash.edu
#SBATCH --mail-type=fail,end,ARRAY_TASKS
# ------------------------------------------

# --- SETUP: Define paths and variables ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NIPYPE_MAX_JOBS=2
export NIPYPE_RESOURCE_MONITOR=1

PROJECT_DIR="/fs04/by84/Brandon/memoria"
SCRATCH_DIR="/fs04/by84_scratch"
if [ ! -d "$SCRATCH_DIR" ] || [ ! -w "$SCRATCH_DIR" ]; then
  SCRATCH_DIR="${PROJECT_DIR}/.scratch"
fi
mkdir -p "$SCRATCH_DIR"

# --- INPUT PATHS ---
BIDS_DIR="${PROJECT_DIR}"
FMRIPREP_DIR="${PROJECT_DIR}/derivatives/fMRIprepNoSbref"

# --- OUTPUT PATH ---
NIBETASERIES_DIR="${PROJECT_DIR}/derivatives/nibetaseries_nosm"
mkdir -p ${NIBETASERIES_DIR}
mkdir -p /fs04/by84/Brandon/memoria/logs/nibetaseries_nosm # Ensure log directory exists

NIBETASERIES_IMG="${PROJECT_DIR}/singularity_images/nibetaseries-v0.6.0.simg" # <-- IMPORTANT: Update this path!

SINGULARITY_WORK_DIR="${SCRATCH_DIR}/nibetaseries_work/${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p ${SINGULARITY_WORK_DIR}

# --- JOB ARRAY LOGIC ---
subject=$(tail -n +2 "${BIDS_DIR}/participants.tsv" | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" | awk '{sub(/^sub-/, "", $1); print $1}')
if [ -z "$subject" ]; then
  echo "ERROR: subject lookup failed for array task ${SLURM_ARRAY_TASK_ID}"
  exit 1
fi

# --- THE NIBETASERIES COMMAND-LINE ARGUMENTS ---
cmd="singularity run --cleanenv \
  --pwd /data \
  -B ${BIDS_DIR}:/data \
  -B ${NIBETASERIES_DIR}:/out \
  -B ${SINGULARITY_WORK_DIR}:/work \
  ${NIBETASERIES_IMG} \
  nibs \
    -w /work \
    -t forgetexp \
    -sp T1w \
    -c csf white_matter trans_x trans_y trans_z rot_x rot_y rot_z a_comp_cor_00 a_comp_cor_01 a_comp_cor_02 a_comp_cor_03 a_comp_cor_04 a_comp_cor_05 \
    -sm 0 \
    --participant-label $subject \
    --nthreads 4 \
    --hrf-model 'glover' \
    /data /data/derivatives/fMRIprepNoSbref /out participant"

# --- JOB EXECUTION ---
echo "--- JOB START ---"
echo "Running NiBetaSeries task ${SLURM_ARRAY_TASK_ID} for sub-${subject}"

# --- START OF EDITED SECTION ---
# This section checks if all sessions for a subject have already been processed.


# --- END OF EDITED SECTION ---

echo "BIDS DIR (on host): ${BIDS_DIR}"
echo "FMRIPREP DIR (on host): ${FMRIPREP_DIR}"
echo "OUTPUT DIR (on host): ${NIBETASERIES_DIR}"
echo "SCRATCH WORK DIR (on host): ${SINGULARITY_WORK_DIR}"
echo "Commandline: $cmd"
echo "---"

echo "Available memory: $(free -h)"
echo "Available disk space: $(df -h $SINGULARITY_WORK_DIR)"

set -o pipefail
eval $cmd 2>&1 | tee ${NIBETASERIES_DIR}/sub-${subject}_nibetaseries.log
exitcode=${PIPESTATUS[0]}

echo "--- JOB END ---"
echo "sub-$subject   ${SLURM_ARRAY_TASK_ID}    $exitcode" \
      >> ${SLURM_JOB_NAME}.${SLURM_ARRAY_JOB_ID}.tsv
echo "Finished task ${SLURM_ARRAY_TASK_ID} for sub-${subject} with exit code $exitcode"

# Clean up the large temporary working directory from scratch space
rm -rf $SINGULARITY_WORK_DIR
echo "Cleaned up work directory: ${SINGULARITY_WORK_DIR}"

exit $exitcode
