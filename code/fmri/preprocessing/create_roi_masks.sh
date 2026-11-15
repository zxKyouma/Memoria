#!/bin/bash

# ---CONFIGURATION---
# This script generates ROI masks from FreeSurfer outputs for multiple subjects.
#
# DEPENDENCIES:
# 1. FreeSurfer (for mri_binarize, mri_label2vol, etc.)
# 2. FSL (for fslmaths to combine hemispheres and ROIs)
#
# INSTRUCTIONS:
# 1. Save this script as 'create_roi_masks.sh'.
# 2. Make it executable: chmod +x create_roi_masks.sh
# 3. IMPORTANT: Verify the BIDS_DIR and FS_DIR paths below.
# 4. Run it from your terminal: ./create_roi_masks.sh
#
# VERSION 2: Added checks to see if masks exist before creating them,
# making the script re-runnable and more efficient.

# Exit script if any command fails
set -e

# --- USER-CONFIGURABLE PATHS ---
BIDS_DIR="${MEMORIA_ROOT}"
FS_DIR="${BIDS_DIR}/derivatives/freesurfer"
OUTPUT_BASE_DIR="${BIDS_DIR}/derivatives/roi_masks"
# --- END OF CONFIGURATION ---

export SUBJECTS_DIR=${FS_DIR}

for SUBJ_DIR in ${FS_DIR}/sub-*/; do
    SUBJ_ID=$(basename "${SUBJ_DIR}")
    if [ "${SUBJ_ID}" == "fsaverage" ]; then continue; fi

    echo "================================================="
    echo "Processing: ${SUBJ_ID}"
    echo "================================================="

    MRI_DIR="${FS_DIR}/${SUBJ_ID}/mri"
    LABEL_DIR="${FS_DIR}/${SUBJ_ID}/label"
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/${SUBJ_ID}/anat"
    mkdir -p "${OUTPUT_DIR}"

    # --- 1. Volumetric ROIs from aparc+aseg.mgz ---
    echo "-> Extracting volumetric ROIs (Hippo, Amyg, PHC, A1, M1, S1)..."
    APARC_ASEG="${MRI_DIR}/aparc+aseg.mgz"
    if [ ! -f "$APARC_ASEG" ]; then echo "ERROR: Cannot find ${APARC_ASEG}."; continue; fi

    # Define mask filenames
    hippo_mask="${OUTPUT_DIR}/${SUBJ_ID}_space-T1w_desc-hippocampus_mask.nii.gz"
    phc_mask="${OUTPUT_DIR}/${SUBJ_ID}_space-T1w_desc-phc_mask.nii.gz"
    amyg_mask="${OUTPUT_DIR}/${SUBJ_ID}_space-T1w_desc-amygdala_mask.nii.gz"
    a1_mask="${OUTPUT_DIR}/${SUBJ_ID}_space-T1w_desc-A1_mask.nii.gz"
    m1_mask="${OUTPUT_DIR}/${SUBJ_ID}_space-T1w_desc-M1_mask.nii.gz"
    s1_mask="${OUTPUT_DIR}/${SUBJ_ID}_space-T1w_desc-S1_mask.nii.gz"

    if [ ! -f "$hippo_mask" ]; then
        echo "   - Creating Hippocampus mask..."
        mri_binarize --i ${APARC_ASEG} --match 17 53 --o "${hippo_mask}"
    else
        echo "   - Hippocampus mask already exists. Skipping."
    fi

    if [ ! -f "$phc_mask" ]; then
        echo "   - Creating PHC mask..."
        mri_binarize --i ${APARC_ASEG} --match 1016 2016 --o "${phc_mask}"
    else
        echo "   - PHC mask already exists. Skipping."
    fi

    if [ ! -f "$amyg_mask" ]; then
        echo "   - Creating Amygdala mask..."
        mri_binarize --i ${APARC_ASEG} --match 18 54 --o "${amyg_mask}"
    else
        echo "   - Amygdala mask already exists. Skipping."
    fi

    # Primary sensory/motor controls from aparc+aseg (FreeSurferColorLUT labels).
    if [ ! -f "$a1_mask" ]; then
        echo "   - Creating A1 (transversetemporal) mask..."
        mri_binarize --i ${APARC_ASEG} --match 1034 2034 --o "${a1_mask}"
    else
        echo "   - A1 mask already exists. Skipping."
    fi

    if [ ! -f "$m1_mask" ]; then
        echo "   - Creating M1 (precentral) mask..."
        mri_binarize --i ${APARC_ASEG} --match 1024 2024 --o "${m1_mask}"
    else
        echo "   - M1 mask already exists. Skipping."
    fi

    if [ ! -f "$s1_mask" ]; then
        echo "   - Creating S1 (postcentral) mask..."
        mri_binarize --i ${APARC_ASEG} --match 1022 2022 --o "${s1_mask}"
    else
        echo "   - S1 mask already exists. Skipping."
    fi

    # --- 2. Extract ALL Labels from Annotation ---
    echo "-> Checking for and extracting all labels from mpm.vpnl.annot..."
    ANNOT_FOUND=true
    # Check only once if we need to run this, to avoid re-running if labels already exist
    if [ ! -f "${LABEL_DIR}/lh.V1.mpm.vpnl.label" ]; then
      for HEMI in lh rh; do
        if [ -f "${LABEL_DIR}/${HEMI}.mpm.vpnl.annot" ]; then
          echo "   - Extracting labels for ${HEMI}..."
          mri_annotation2label --subject ${SUBJ_ID} --hemi ${HEMI} --annotation mpm.vpnl --outdir ${LABEL_DIR}
        else
          echo "WARNING: Annotation file ${LABEL_DIR}/${HEMI}.mpm.vpnl.annot not found."
          ANNOT_FOUND=false
        fi
      done
    else
        echo "   - Labels appear to be extracted already. Skipping."
    fi


    # --- 3. Surface-based ROIs ---
    echo "-> Projecting surface labels to volume..."
    T1_MGZ="${MRI_DIR}/T1.mgz"
    if [ ! -f "$T1_MGZ" ]; then echo "ERROR: Cannot find ${T1_MGZ}."; continue; fi

    find_label_file() {
        local hemi=$1; local roi_base_name=$2
        if [ -f "${LABEL_DIR}/${hemi}.${roi_base_name}_exvivo.label" ]; then echo "${LABEL_DIR}/${hemi}.${roi_base_name}_exvivo.label"
        elif [ -f "${LABEL_DIR}/${hemi}.${roi_base_name}.mpm.vpnl.label" ]; then echo "${LABEL_DIR}/${hemi}.${roi_base_name}.mpm.vpnl.label"
        elif [ -f "${LABEL_DIR}/${hemi}.${roi_base_name}.label" ]; then echo "${LABEL_DIR}/${hemi}.${roi_base_name}.label"
        else echo "NOT_FOUND"; fi
    }

    create_combined_mask() {
        local roi_name=$1; local output_file_path=$2

        # *** NEW: Check if the final mask already exists ***
        if [ -f "$output_file_path" ]; then
            echo "   - Bilateral ${roi_name} mask already exists. Skipping."
            return 0
        fi

        local lh_label_file=$(find_label_file "lh" "${roi_name}"); local rh_label_file=$(find_label_file "rh" "${roi_name}")

        if [ "$lh_label_file" == "NOT_FOUND" ] || [ "$rh_label_file" == "NOT_FOUND" ]; then
            echo "   - WARNING: Label files for ${roi_name} not found. Skipping creation."
            return 1
        fi

        local tmp_lh_mask="${OUTPUT_DIR}/tmp.lh.${roi_name}.nii.gz"; local tmp_rh_mask="${OUTPUT_DIR}/tmp.rh.${roi_name}.nii.gz"

        mri_label2vol --label "${lh_label_file}" --temp "${T1_MGZ}" --subject "${SUBJ_ID}" --hemi lh --o "${tmp_lh_mask}" --proj frac 0 1 0.1 --regheader "${T1_MGZ}"
        mri_label2vol --label "${rh_label_file}" --temp "${T1_MGZ}" --subject "${SUBJ_ID}" --hemi rh --o "${tmp_rh_mask}" --proj frac 0 1 0.1 --regheader "${T1_MGZ}"

        fslmaths "${tmp_lh_mask}" -add "${tmp_rh_mask}" -bin "${output_file_path}"
        rm "${tmp_lh_mask}" "${tmp_rh_mask}"
        echo "   - Created bilateral ${roi_name} mask."
        return 0
    }

    create_combined_mask "V1" "${OUTPUT_DIR}/${SUBJ_ID}_space-T1w_desc-V1_mask.nii.gz"
    create_combined_mask "hOc4v" "${OUTPUT_DIR}/${SUBJ_ID}_space-T1w_desc-hV4_mask.nii.gz"
    create_combined_mask "MT" "${OUTPUT_DIR}/${SUBJ_ID}_space-T1w_desc-MT_mask.nii.gz"

    # --- 4. Create composite LO/IT ROI ---
    if ! $ANNOT_FOUND; then echo "-> Skipping LO/IT mask creation due to missing annotation file."; continue; fi
    echo "-> Creating composite LO/IT mask..."

    LOIT_COMPONENT_ROIS=("LO1" "LO2" "PIT" "FG1" "FG2" "FG3" "FG4")
    for roi in "${LOIT_COMPONENT_ROIS[@]}"; do
        create_combined_mask "${roi}" "${OUTPUT_DIR}/${SUBJ_ID}_space-T1w_desc-${roi}_mask.nii.gz" || true
    done

    # Create composite Fusiform mask
    fus_mask_file="${OUTPUT_DIR}/${SUBJ_ID}_space-T1w_desc-FUS_mask.nii.gz"
    if [ ! -f "$fus_mask_file" ]; then
        FUS_MASK_LIST=()
        for roi in FG1 FG2 FG3 FG4; do
            mask_file="${OUTPUT_DIR}/${SUBJ_ID}_space-T1w_desc-${roi}_mask.nii.gz"
            [ -f "$mask_file" ] && FUS_MASK_LIST+=("$mask_file")
        done
        if [ ${#FUS_MASK_LIST[@]} -gt 0 ]; then
            echo "   - Combining Fusiform (FG1-4) components..."
            fslmaths "${FUS_MASK_LIST[0]}" $(for m in "${FUS_MASK_LIST[@]:1}"; do echo "-add $m"; done) -bin "$fus_mask_file"
        fi
    else
        echo "   - Composite FUS mask already exists. Skipping."
    fi

    # Create composite LOIT mask
    loit_mask_file="${OUTPUT_DIR}/${SUBJ_ID}_space-T1w_desc-LOIT_mask.nii.gz"
    if [ ! -f "$loit_mask_file" ]; then
        LOIT_MASK_LIST=()
        for roi_desc in LO1 LO2 PIT FUS; do
            mask_file="${OUTPUT_DIR}/${SUBJ_ID}_space-T1w_desc-${roi_desc}_mask.nii.gz"
            [ -f "$mask_file" ] && LOIT_MASK_LIST+=("$mask_file")
        done
        if [ ${#LOIT_MASK_LIST[@]} -gt 0 ]; then
            echo "   - Combining all components into final LOIT mask..."
            fslmaths "${LOIT_MASK_LIST[0]}" $(for m in "${LOIT_MASK_LIST[@]:1}"; do echo "-add $m"; done) -bin "$loit_mask_file"
        fi
    else
        echo "   - Composite LOIT mask already exists. Skipping."
    fi
done

echo "================================================="
echo "All subjects processed."
echo "Generated masks are located in: ${OUTPUT_BASE_DIR}"
echo "Next step: Run the resample_masks.py script."
echo "================================================="
