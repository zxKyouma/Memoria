#!/bin/bash
#SBATCH -J fmriprep-array # A better job name for a job array
#SBATCH --time=24:00:00
#SBATCH -n 1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=16G   # Total memory = 12 * 16 = 192 GB
# Outputs ----------------------------------
# Log files will be saved in your project directory, which is fine.
#SBATCH -o /home/zlow/by84/Brandon/memoria/logs/%x-%A-%a.out
#SBATCH -e /home/zlow/by84/Brandon/memoria/logs/%x-%A-%a.err
#SBATCH --mail-user=zlow0014@student.monash.edu
#SBATCH --mail-type=fail,end,ARRAY_TASKS
# ------------------------------------------

PROJECT_DIR="/home/zlow/by84/Brandon/memoria"

SCRATCH_DIR="/home/zlow/by84_scratch"

BIDS_DIR="${PROJECT_DIR}"
DERIVS_DIR="${PROJECT_DIR}/derivatives/fMRIprepNoSbref"
FREESURFER_DIR="${PROJECT_DIR}/derivatives/freesurfer"
mkdir -p ${DERIVS_DIR}
mkdir -p ${FREESURFER_DIR}

TEMPLATEFLOW_HOST_HOME=${SCRATCH_DIR}/.cache/templateflow
FMRIPREP_HOST_CACHE=${SCRATCH_DIR}/.cache/fmriprep
SINGULARITY_WORK_DIR="${SCRATCH_DIR}/fmriprep_work/${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}" # Unique work dir for each subject
mkdir -p ${TEMPLATEFLOW_HOST_HOME} ${FMRIPREP_HOST_CACHE} ${SINGULARITY_WORK_DIR}

FMRIPREP_IMG="$PROJECT_DIR/singularity_images/fmriprep-23-2-0.simg"

FS_LICENSE_ON_HOST="/home/zlow/by84/Brandon/freesurfer_license.txt"
# A simple path where we will make it available INSIDE the container
FS_LICENSE_IN_CONTAINER="/fs_license.txt"

export SINGULARITYENV_TEMPLATEFLOW_HOME=/templateflow_cache


SINGULARITY_CMD="singularity run --cleanenv \
  -B ${BIDS_DIR}:/bids:ro \
  -B ${DERIVS_DIR}:/output \
  -B ${FREESURFER_DIR}:/fsdir \
  -B ${SINGULARITY_WORK_DIR}:/work \
  -B ${TEMPLATEFLOW_HOST_HOME}:${SINGULARITYENV_TEMPLATEFLOW_HOME} \
  -B ${FS_LICENSE_ON_HOST}:${FS_LICENSE_IN_CONTAINER}:ro \
  ${FMRIPREP_IMG}"

subject=$( sed -n -E "$((${SLURM_ARRAY_TASK_ID} + 1))s/sub-(\S*)\>.*/\1/gp" ${BIDS_DIR}/participants.tsv )

find ${FREESURFER_DIR}/sub-$subject/ -name "*IsRunning*" -type f -delete

cmd="${SINGULARITY_CMD} /bids /output participant \
  --participant-label $subject \
  -w /work/ \
  --fs-subjects-dir /fsdir \
  --fs-license-file ${FS_LICENSE_IN_CONTAINER} \
  -vv \
  --omp-nthreads 12 \
  --nthreads 12 \
  --mem_mb 180000 \
  --bold2t1w-init register \
  --output-spaces MNI152NLin2009cAsym anat fsnative \
  --use-aroma \
  --ignore sbref t2w"

echo "--- JOB START ---"
echo "Running task ${SLURM_ARRAY_TASK_ID} for sub-${subject}"
echo "BIDS DIR (on host): ${BIDS_DIR}"
echo "DERIVS DIR (on host): ${DERIVS_DIR}"
echo "SCRATCH WORK DIR (on host): ${SINGULARITY_WORK_DIR}"
echo "Commandline: $cmd"
echo "---"

eval $cmd
exitcode=$?

echo "--- JOB END ---"
# Output results to a summary file for this job array
echo "sub-$subject   ${SLURM_ARRAY_TASK_ID}    $exitcode" \
      >> ${SLURM_JOB_NAME}.${SLURM_ARRAY_JOB_ID}.tsv
echo "Finished task ${SLURM_ARRAY_TASK_ID} for sub-${subject} with exit code $exitcode"

# Clean up the large temporary working directory from scratch space
rm -rf $SINGULARITY_WORK_DIR
echo "Cleaned up work directory: ${SINGULARITY_WORK_DIR}"

exit $exitcode
