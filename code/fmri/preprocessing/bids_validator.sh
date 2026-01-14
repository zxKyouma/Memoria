# --- BIDS Validator Diagnostic ---

# Define the paths to your data and singularity image
BIDS_DIR="/home/zlow/by84/Brandon/memoria"
FMRIPREP_IMG="${BIDS_DIR}/singularity_images/fmriprep-23-2-0.simg"

# Use 'singularity exec' to run a command INSIDE the container
# This will run the validator on your Bids directory and print all errors/warnings
singularity exec \
  -B ${BIDS_DIR}:/bids \
  ${FMRIPREP_IMG} \
  bids-validator /bids --verbose
