import os
import glob
import pandas as pd
import numpy as np # <<< Import numpy for random sampling
from nilearn.image import load_img
from nilearn.maskers import NiftiMasker
import skdim

# --- USER CONFIGURATION ---
BIDS_DIR = os.environ.get("MEMORIA_ROOT", ".")
SUBJECT_ID = "62428"
SESSION_ID = "1"
TASK_NAME = "forgetexp"

ROI_NAMES_IN_ORDER = ["V1", "hV4", "LOIT", "phc", "hippocampus"]

# <<< Add a random seed for reproducibility of the voxel subsampling
RANDOM_SEED = 42

# --- Path Templates (These remain the same) ---
BETA_FILES_TEMPLATE = os.path.join(
    BIDS_DIR, "derivatives", "nibetaseries_nosm", "nibetaseries",
    f"sub-{SUBJECT_ID}", f"ses-{SESSION_ID}", "func",
    f"sub-{SUBJECT_ID}_ses-{SESSION_ID}_task-{TASK_NAME}_acq-epfid2d1210_space-T1w_desc-Feedback*_betaseries.nii.gz"
)
ROI_MASK_TEMPLATE = os.path.join(
    BIDS_DIR, "derivatives", "roi_masks", f"sub-{SUBJECT_ID}", "anat",
    f"sub-{SUBJECT_ID}_space-T1w_desc-{{roi_name}}_resampled-betaseries_mask.nii.gz"
)
# --- END OF CONFIGURATION ---


def main():
    """
    Calculates intrinsic dimensionality for a single subject across specified ROIs,
    controlling for the number of voxels used in the calculation.
    """
    # <<< Set the random seed to make sure the random voxel selection is the same every time
    np.random.seed(RANDOM_SEED)

    print(f"--- Starting Intrinsic Dimensionality Analysis for sub-{SUBJECT_ID} (Voxel-Controlled) ---")
    
    # Find all beta-series files
    print(f"Searching for ENCODING beta-series files using pattern:\n  {BETA_FILES_TEMPLATE}")
    beta_files_list = sorted(glob.glob(BETA_FILES_TEMPLATE))

    if not beta_files_list:
        print("\n!!! ERROR: No encoding beta-series files found. Please check your BETA_FILES_TEMPLATE.")
        return
        
    print(f"\nFound {len(beta_files_list)} ENCODING beta-series files for the analysis.")

    # <<< STAGE 1: INSPECTION PASS - Find the minimum number of voxels across all ROIs
    print("\n--- Stage 1: Inspecting ROIs to find minimum voxel count ---")
    roi_voxel_counts = {}
    for roi_name in ROI_NAMES_IN_ORDER:
        roi_mask_path = ROI_MASK_TEMPLATE.format(roi_name=roi_name)
        if not os.path.exists(roi_mask_path):
            print(f"  - WARNING: Mask file for {roi_name} not found, skipping inspection.")
            continue
        
        try:
            # Create a masker and fit it to just the first beta file to quickly get voxel count
            masker = NiftiMasker(mask_img=roi_mask_path)
            # Use fit_transform on one image to get the shape
            n_voxels = masker.fit_transform(beta_files_list[0]).shape[1]
            if n_voxels > 0:
                roi_voxel_counts[roi_name] = n_voxels
                print(f"  - {roi_name}: Found {n_voxels} voxels.")
            else:
                print(f"  - WARNING: Mask for {roi_name} is empty (0 voxels).")
        except Exception as e:
            print(f"  - FAILED to inspect {roi_name}. Error: {e}")

    if not roi_voxel_counts:
        print("\n!!! ERROR: Could not find any valid ROIs. Exiting.")
        return

    min_voxel_count = min(roi_voxel_counts.values())
    print(f"\n>>> Voxel count for analysis will be fixed to the minimum: {min_voxel_count}")

    # <<< STAGE 2: ANALYSIS PASS - Calculate ID using the fixed number of voxels
    print("\n--- Stage 2: Running analysis with controlled voxel count ---")
    results = []
    for roi_name in ROI_NAMES_IN_ORDER:
        print(f"\nProcessing ROI: {roi_name}")
        
        # Check if this ROI was valid in the first pass
        if roi_name not in roi_voxel_counts:
            print("  - Skipping, as it was not found or was empty in the inspection pass.")
            continue

        roi_mask_path = ROI_MASK_TEMPLATE.format(roi_name=roi_name)
        masker = NiftiMasker(mask_img=roi_mask_path, standardize=False, memory="nilearn_cache", verbose=0)

        try:
            # Extract data from ALL voxels first
            full_voxel_data = masker.fit_transform(beta_files_list)
            n_samples, n_original_features = full_voxel_data.shape
            print(f"  - Original data shape: (samples={n_samples}, voxels={n_original_features})")
            
            # <<< The key subsampling step
            print(f"  - Randomly subsampling to {min_voxel_count} voxels for fair comparison.")
            # Create an array of indices [0, 1, 2, ..., n_original_features-1]
            voxel_indices = np.arange(n_original_features)
            # Choose 'min_voxel_count' indices from that array without replacement
            subsampled_indices = np.random.choice(voxel_indices, size=min_voxel_count, replace=False)
            # Select only the columns (voxels) corresponding to the chosen indices
            subsampled_voxel_data = full_voxel_data[:, subsampled_indices]
            
            # Calculate ID on the SUBSAMPLED data
            id_estimator = skdim.id.TwoNN().fit(subsampled_voxel_data)
            intrinsic_dim = id_estimator.dimension_
            print(f"  - Estimated Intrinsic Dimensionality (ID): {intrinsic_dim:.2f}")

            # Store the results, including both original and used voxel counts
            results.append({
                "ROI": roi_name,
                "Intrinsic_Dim": intrinsic_dim,
                "Num_Trials": n_samples,
                "Original_Voxels": n_original_features,
                "Used_Voxels": min_voxel_count
            })

        except Exception as e:
            print(f"  - FAILED to process {roi_name}. Error: {e}")

    # Display final results
    if results:
        results_df = pd.DataFrame(results)
        results_df['ROI'] = pd.Categorical(results_df['ROI'], categories=ROI_NAMES_IN_ORDER, ordered=True)
        results_df = results_df.sort_values('ROI')
        
        print("\n\n--- FINAL RESULTS (Voxel-Controlled) ---")
        print(f"Subject: {SUBJECT_ID} | Random Seed: {RANDOM_SEED}")
        print(results_df.to_string(index=False))
    else:
        print("\n--- No results were generated. Please check paths and files. ---")


if __name__ == '__main__':
    main()
