#!/usr/bin/env python3

# ---CONFIGURATION---
# This script resamples specific, generated ROI masks to match the voxel grid of
# reference beta-series files from the nibetaseries derivative for specific sessions.
# This version is robust and prevents re-processing its own output files.

import os
import glob
from nilearn.image import resample_to_img

# --- USER-CONFIGURABLE PATHS AND SUFFIXES ---
BIDS_DIR = os.environ.get("MEMORIA_ROOT", ".")
ROI_DIR = os.path.join(BIDS_DIR, "derivatives", "roi_masks")

# Define the specific ROI descriptions ('desc-<label>') you want to process.
TARGET_ROIS = [
    "LOIT", "FUS", "V1", "hV4", "hippocampus", "amygdala", "phc",
    "A1", "M1", "S1", "MT"
]
SESSIONS = ["ses-1", "ses-2"]
OUT_SUFFIX = "_resampled-betaseries_mask.nii.gz"

# Template for finding the session-specific reference functional image
REFERENCE_IMAGE_TEMPLATE = os.path.join(
    BIDS_DIR, "derivatives", "nibetaseries", "nibetaseries",
    "{subject_id}", "{session}", "func",
    "{subject_id}_{session}_*_space-T1w_*_betaseries.nii.gz"
)
# --- END OF CONFIGURATION ---

def main():
    print("--- Starting Session-Specific ROI Resampling ---")

    subject_dirs = sorted(glob.glob(os.path.join(ROI_DIR, "sub-*")))
    if not subject_dirs:
        print(f"Error: No subject directories found in {ROI_DIR}")
        return

    for subj_dir in subject_dirs:
        subject_id = os.path.basename(subj_dir)
        anat_dir = os.path.join(subj_dir, "anat")
        print(f"\nProcessing: {subject_id}")

        for roi_desc in TARGET_ROIS:
            # === THE KEY TO PREVENTING DUPLICATES ===
            # We construct the *exact* filename of the original source mask.
            # This will ONLY match files like "sub-XXXX_space-T1w_desc-phc_mask.nii.gz".
            # It will NEVER match a file that already contains "_resampled-betaseries_".
            source_mask_path = os.path.join(
                anat_dir,
                f"{subject_id}_space-T1w_desc-{roi_desc}_mask.nii.gz"
            )

            if not os.path.exists(source_mask_path):
                print(f"  - INFO: Source mask for '{roi_desc}' not found. Skipping.")
                continue

            print(f"  - Found source mask: {os.path.basename(source_mask_path)}")

            for session in SESSIONS:
                ref_image_path_pattern = REFERENCE_IMAGE_TEMPLATE.format(
                    subject_id=subject_id, session=session
                )
                ref_image_list = sorted(glob.glob(ref_image_path_pattern))

                if not ref_image_list:
                    print(f"    - WARNING: Ref image not found for {session}. Skipping.")
                    continue
                
                reference_image = ref_image_list[0]

                # Construct the new, session-specific output filename
                output_filename = f"{subject_id}_{session}_space-T1w_desc-{roi_desc}{OUT_SUFFIX}"
                output_path = os.path.join(anat_dir, output_filename)

                if os.path.exists(output_path):
                    print(f"    - Skipping {session}, already exists: {os.path.basename(output_path)}")
                    continue

                try:
                    print(f"    - Resampling for {session}...")
                    resampled_mask = resample_to_img(source_mask_path, reference_image, interpolation='nearest')
                    resampled_mask.to_filename(output_path)
                    print(f"    - Created: {os.path.basename(output_path)}")
                except Exception as e:
                    print(f"    - FAILED to resample for {session}. Error: {e}")

    print("\n--- Resampling complete! ---")

if __name__ == '__main__':
    main()
