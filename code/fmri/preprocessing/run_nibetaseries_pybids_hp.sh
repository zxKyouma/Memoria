#!/bin/bash
#SBATCH -J nibetaseries-hp-array
#SBATCH --time=72:00:00
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -o logs/nibetaseries_hp/%x-%A-%a.out
#SBATCH -e logs/nibetaseries_hp/%x-%A-%a.err

# --- SETUP: Define paths and variables ---
PROJECT_DIR="${PROJECT_DIR:-${MEMORIA_ROOT:-$PWD}}"
SCRATCH_DIR="${SCRATCH_DIR:-/tmp}"
if ! mkdir -p "${SCRATCH_DIR}" 2>/dev/null; then
  SCRATCH_DIR="${PROJECT_DIR}/.scratch"
  mkdir -p "${SCRATCH_DIR}"
fi
cd "${PROJECT_DIR}" || exit 1

# --- INPUT PATHS ---
BIDS_DIR="${PROJECT_DIR}"
FMRIPREP_DIR="${PROJECT_DIR}/derivatives/fMRIprepNoSbref"

# --- OUTPUT PATH ---
NIBETASERIES_DIR="${PROJECT_DIR}/derivatives/nibetaseries_hp"
mkdir -p ${NIBETASERIES_DIR}
mkdir -p ${MEMORIA_ROOT}/logs/nibetaseries_hp

NIBETASERIES_IMG="${PROJECT_DIR}/singularity_images/nibetaseries-v0.6.0.simg"

mkdir -p "${SCRATCH_DIR}/nibetaseries_work"
SINGULARITY_WORK_DIR="${SCRATCH_DIR}/nibetaseries_work/${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "${SINGULARITY_WORK_DIR}"

# --- JOB ARRAY LOGIC ---
subject=$(tail -n +2 "${BIDS_DIR}/participants.tsv" | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" | awk '{sub(/^sub-/, "", $1); print $1}')
if [ -z "$subject" ]; then
  echo "ERROR: subject lookup failed for array task ${SLURM_ARRAY_TASK_ID}"
  exit 1
fi

# --- FIND COSINE REGRESSORS (HIGH-PASS) ---
mapfile -t confounds_files < <(ls "${FMRIPREP_DIR}/sub-${subject}"/ses-*/func/*task-forgetexp*_desc-confounds_regressors.tsv 2>/dev/null)
if [ ${#confounds_files[@]} -eq 0 ]; then
  echo "ERROR: confounds_regressors.tsv not found for sub-${subject}"
  exit 1
fi

# Use the minimum cosine set across runs to avoid missing-column errors.
min_cosines=999
for f in "${confounds_files[@]}"; do
  n=$(awk -F'\t' 'NR==1 {c=0; for(i=1;i<=NF;i++) if($i ~ /^cosine[0-9]+$/) c++; print c; exit}' "$f")
  if [ "$n" -lt "$min_cosines" ]; then
    min_cosines="$n"
  fi
done

COSINE_COLS=""
if [ "$min_cosines" -gt 0 ]; then
  for i in $(seq 0 $((min_cosines - 1))); do
    COSINE_COLS+="cosine$(printf '%02d' "$i") "
  done
fi

BASE_CONFOUNDS="csf white_matter trans_x trans_y trans_z rot_x rot_y rot_z a_comp_cor_00 a_comp_cor_01 a_comp_cor_02 a_comp_cor_03 a_comp_cor_04 a_comp_cor_05"
CONFOUND_LIST="${BASE_CONFOUNDS} ${COSINE_COLS}"

echo "Using confounds: ${CONFOUND_LIST}"

# --- THE NIBETASERIES COMMAND-LINE ARGUMENTS ---
cmd="singularity run --cleanenv \
  --pwd /data \
  -B ${BIDS_DIR}:/data \
  -B ${FMRIPREP_DIR}:/fmriprep \
  -B ${NIBETASERIES_DIR}:/out \
  -B ${SINGULARITY_WORK_DIR}:/work \
  ${NIBETASERIES_IMG} \
  nibs \
    -w /work \
    -t forgetexp \
    -sp T1w \
    -c ${CONFOUND_LIST} \
    --bids-database-dir /work/.pybids \
    --reset-database \
    --participant-label $subject \
    --nthreads 8 \
    --hrf-model 'glover' \
    /data /fmriprep /out participant"

# --- JOB EXECUTION ---
echo "--- JOB START ---"
echo "Running NiBetaSeries (high-pass via cosine regressors) task ${SLURM_ARRAY_TASK_ID} for sub-${subject}"
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
