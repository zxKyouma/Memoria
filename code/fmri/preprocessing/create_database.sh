#!/bin/bash
#SBATCH -J nibs-db-builder
#SBATCH -n 1
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH -o /fs04/by84/Brandon/memoria/logs/nibetaseries/%x-%j.out
#SBATCH -e /fs04/by84/Brandon/memoria/logs/nibetaseries/%x-%j.err
#SBATCH --mail-user=zlow0014@student.monash.edu
#SBATCH --mail-type=end

echo "--- JOB START: BIDS DATABASE GENERATION (v0.6.0 compatible) ---"

# --- Define paths ---
PROJECT_DIR="/fs04/by84/Brandon/memoria"
FMRIPREP_DIR="${PROJECT_DIR}/derivatives/fMRIprepNoSbref"
NIBETASERIES_IMG="${PROJECT_DIR}/singularity_images/nibetaseries-v0.6.0.simg"

echo "Building PyBIDS database for ${PROJECT_DIR} and its derivatives..."

# This command runs a short Python script inside the container.
# The BIDSLayout command will automatically find and index the BIDS dataset
# and the specified fMRIPrep derivatives, creating the database in
# /data/.pybids/, which corresponds to ${PROJECT_DIR}/.pybids/ on your host.
singularity exec --cleanenv \
  -B ${PROJECT_DIR}:/data \
  -B ${FMRIPREP_DIR}:/fmriprep \
  ${NIBETASERIES_IMG} \
  python -c "from bids.layout import BIDSLayout; BIDSLayout('/data', derivatives='/fmriprep', validate=True)"

exitcode=$?
if [ $exitcode -eq 0 ]; then
    echo "?  Database built successfully in ${PROJECT_DIR}/.pybids/"
else
    echo "?  Database generation failed with exit code $exitcode."
fi

echo "--- JOB END ---"

exit $exitcode
