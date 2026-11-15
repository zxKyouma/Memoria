import os
import glob
import argparse
import re
import pandas as pd
import numpy as np
from nilearn.image import load_img, index_img
from nilearn.maskers import NiftiMasker
import skdim
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- USER CONFIGURATION ---
BIDS_DIR = os.environ.get("MEMORIA_ROOT", ".")
TASK_NAME = "forgetexp"
# TRIAL_TYPES_DEFAULT = ["Enc", "Face", "Recall", "Feedback"] # ses-1 forget-exp
# TRIAL_TYPES_DEFAULT = ["recog", "faceRecog", "ret", "retRating", "retRatingFeedback", "recogRating", "recogFeedback", "recogRatingFeedback"] # ses-2 forget-exp
TRIAL_TYPES_DEFAULT = ["ret", "retFeedback", "retRating", "retRatingFeedback"]
ROI_NAMES_DEFAULT = ["V1", "hV4", "LOIT", "phc", "hippocampus"]
RANDOM_SEED = 42
OUTPUT_DIR_DEFAULT = "./id_analysis_results_by_condition_ses-2_forgetexp"
# --- END OF CONFIGURATION ---

def load_single_volume_imgs(paths):
    """Load beta-series images and squeeze 4D single-volume files to 3D."""
    imgs = []
    for path in paths:
        img = load_img(path)
        if len(img.shape) == 4 and img.shape[-1] == 1:
            img = index_img(img, 0)
        imgs.append(img)
    return imgs

def find_subjects(nibetaseries_dir):
    """Finds all potential subject IDs from the nibetaseries directory."""
    subject_dirs = glob.glob(os.path.join(nibetaseries_dir, "sub-*"))
    subject_ids = [os.path.basename(d).split('-')[1] for d in subject_dirs if os.path.isdir(d)]
    if not subject_ids:
        # This error is now a warning, as the pre-flight check will handle the final error.
        print(f"Warning: No subject directories found in {nibetaseries_dir}.")
    return sorted(subject_ids)


def pre_analysis_file_check(all_subject_ids, roi_names, trial_types, session_id="2"):
    """
    Checks for the existence of all required files for each subject before analysis.
    
    Returns a list of subject IDs that have all necessary files.
    """
    print("\n--- Running Pre-Analysis File Check ---")
    valid_subjects = []
    excluded_subjects = {}

    for subject_id in all_subject_ids:
        missing_files_for_subject = []
        
        # 1. Check for all required ROI masks
        for roi_name in roi_names:
            roi_mask_path = os.path.join(
                BIDS_DIR, "derivatives", "roi_masks", f"sub-{subject_id}", "anat",
                f"sub-{subject_id}_ses-{session_id}_space-T1w_desc-{roi_name}_resampled-betaseries_mask.nii.gz"
            )
            if not os.path.exists(roi_mask_path):
                missing_files_for_subject.append(f"ROI mask: {os.path.basename(roi_mask_path)}")

        # 2. Check for beta series files for each condition
        for trial_type in trial_types:
            beta_files_template = os.path.join(
                BIDS_DIR, "derivatives", "nibetaseries_nosm", "nibetaseries",
                f"sub-{subject_id}", f"ses-{session_id}", "func",
                f"sub-{subject_id}_ses-{session_id}_task-{TASK_NAME}_acq-*_space-T1w_desc-{trial_type}*_betaseries.nii.gz"
            )
            if not glob.glob(beta_files_template):
                missing_files_for_subject.append(f"Beta series for condition: {trial_type}")

        if not missing_files_for_subject:
            valid_subjects.append(subject_id)
        else:
            excluded_subjects[subject_id] = missing_files_for_subject
    
    # --- Reporting Results ---
    print(f"Found {len(all_subject_ids)} potential subjects initially.")
    
    if excluded_subjects:
        print(f"\nExcluding {len(excluded_subjects)} subjects due to missing files:")
        for sub_id, missing in excluded_subjects.items():
            print(f"  - Subject {sub_id}:")
            for reason in missing:
                print(f"    - Missing: {reason}")

    print("\n--- File Check Summary ---")
    print(f"Total subjects ready for analysis: {len(valid_subjects)}")
    if valid_subjects:
        print(f"Proceeding with subjects: {valid_subjects}")
    else:
        print("No subjects have all the required files. Aborting analysis.")
        
    return valid_subjects


def calculate_id_for_condition(subject_id, trial_type, roi_names, session_id="2"):
    """
    Calculates voxel-controlled intrinsic dimensionality for a single subject and a single trial type (condition).
    """
    np.random.seed(RANDOM_SEED)

    beta_files_template = os.path.join(
        BIDS_DIR, "derivatives", "nibetaseries_nosm", "nibetaseries",
        f"sub-{subject_id}", f"ses-{session_id}", "func",
        f"sub-{subject_id}_ses-{session_id}_task-{TASK_NAME}_acq-*_space-T1w_desc-{trial_type}*_betaseries.nii.gz"
    )
    roi_mask_template = os.path.join(
        BIDS_DIR, "derivatives", "roi_masks", f"sub-{subject_id}", "anat",
        f"sub-{subject_id}_ses-{session_id}_space-T1w_desc-{{roi_name}}_resampled-betaseries_mask.nii.gz"
    )

    beta_files_list = sorted(glob.glob(beta_files_template))
    # Drop NA aggregate files (e.g., desc-retNA_betaseries) that break nilearn input stacking.
    beta_pattern = re.compile(rf"_desc-{re.escape(trial_type)}(\d+)_betaseries\.nii\.gz$")
    beta_files_list = [p for p in beta_files_list if beta_pattern.search(os.path.basename(p))]
    if not beta_files_list:
        # This check is now redundant due to the pre-flight check, but is good for safety.
        return []
    beta_imgs = load_single_volume_imgs(beta_files_list)

    roi_voxel_counts = {}
    for roi_name in roi_names:
        roi_mask_path = roi_mask_template.format(roi_name=roi_name)
        # We know the path exists from the pre-flight check
        try:
            masker = NiftiMasker(mask_img=roi_mask_path)
            # Use only the first beta file to quickly get voxel count
            n_voxels = masker.fit_transform(beta_imgs[0]).shape[1]
            if n_voxels > 0:
                roi_voxel_counts[roi_name] = n_voxels
        except Exception:
            # This might catch issues with the NIfTI file itself, even if it exists
            continue

    if not roi_voxel_counts:
        return []

    min_voxel_count = min(roi_voxel_counts.values())

    run_results = []
    for roi_name in roi_names:
        if roi_name not in roi_voxel_counts:
            continue

        roi_mask_path = roi_mask_template.format(roi_name=roi_name)
        masker = NiftiMasker(mask_img=roi_mask_path, standardize=False)

        try:
            full_voxel_data = masker.fit_transform(beta_imgs)
            n_samples, n_original_features = full_voxel_data.shape

            voxel_indices = np.arange(n_original_features)
            subsampled_indices = np.random.choice(voxel_indices, size=min_voxel_count, replace=False)
            subsampled_voxel_data = full_voxel_data[:, subsampled_indices]

            id_estimator = skdim.id.TwoNN().fit(subsampled_voxel_data)
            intrinsic_dim = id_estimator.dimension_

            run_results.append({
                "Subject": subject_id,
                "Condition": trial_type,
                "ROI": roi_name,
                "Intrinsic_Dim": intrinsic_dim,
                "Num_Trials": n_samples,
                "Original_Voxels": n_original_features,
                "Used_Voxels": min_voxel_count
            })
        except Exception as e:
            print(f"\n  - ERROR processing sub-{subject_id}, cond-{trial_type}, ROI-{roi_name}. Error: {e}")

    return run_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute intrinsic dimensionality by condition across ROIs."
    )
    parser.add_argument(
        "--roi-list",
        nargs="+",
        default=ROI_NAMES_DEFAULT,
        help="List of ROI names to analyze (default: ventral stream ROIs).",
    )
    parser.add_argument(
        "--trial-types",
        nargs="+",
        default=TRIAL_TYPES_DEFAULT,
        help="Trial types/conditions to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR_DEFAULT,
        help="Directory to write output CSVs and plots.",
    )
    parser.add_argument(
        "--session",
        default="2",
        help="BIDS session label (default: 2).",
    )
    return parser.parse_args()


def main():
    """
    Main function to run the group-level intrinsic dimensionality analysis across conditions.
    """
    args = parse_args()
    roi_names = args.roi_list
    trial_types = args.trial_types
    output_dir = args.output_dir
    session_id = args.session
    os.makedirs(output_dir, exist_ok=True)

    print("--- Starting Group-Level Intrinsic Dimensionality Analysis (by Condition) ---")
    print(f"Analyzing BIDS Task: '{TASK_NAME}'")
    print(f"Iterating over Conditions: {trial_types}")
    print(f"ROIs: {roi_names}")

    # --- Step 1: Find all potential subjects ---
    nibetaseries_base_dir = os.path.join(BIDS_DIR, "derivatives", "nibetaseries_nosm", "nibetaseries")
    all_subjects = find_subjects(nibetaseries_base_dir)
    if not all_subjects:
        print("CRITICAL: No subject directories found at all. Please check BIDS_DIR path.")
        return

    # --- Step 2: Run the pre-analysis file check to get a list of valid subjects ---
    valid_subject_list = pre_analysis_file_check(
        all_subjects, roi_names, trial_types, session_id=session_id
    )

    # --- Step 3: Proceed with analysis only if there are valid subjects ---
    if not valid_subject_list:
        return # Stop execution if no subjects are valid

    print("\n--- Starting Main Analysis on Validated Subjects ---")
    all_results = []
    
    # Use the validated list for the main analysis loop
    for subject_id in tqdm(valid_subject_list, desc="Processing Subjects"):
        for trial_type in trial_types:
            results_for_run = calculate_id_for_condition(
                subject_id, trial_type, roi_names, session_id=session_id
            )
            if results_for_run:
                all_results.extend(results_for_run)

    if not all_results:
        print("\n\n!!! ANALYSIS FAILED: No results were generated despite passing file check. Check for errors during processing.")
        return

    # --- AGGREGATION ---
    print("\n--- Analysis Complete. Aggregating all data. ---")
    results_df = pd.DataFrame(all_results)
    results_df['ROI'] = pd.Categorical(results_df['ROI'], categories=roi_names, ordered=True)
    results_df['Condition'] = pd.Categorical(results_df['Condition'], categories=trial_types, ordered=True)
    results_df = results_df.sort_values(['Subject', 'Condition', 'ROI'])
    
    # --- SAVE RESULTS (CRITICAL STEP BEFORE VISUALIZATION) ---
    print("\n--- Saving Computed Results to CSV Files ---")
    
    full_results_path = os.path.join(output_dir, "group_id_full_results_by_condition.csv")
    results_df.to_csv(full_results_path, index=False)
    print(f"Full, detailed results saved to: {full_results_path}")

    group_stats = results_df.groupby(['Condition', 'ROI'])['Intrinsic_Dim'].agg(
        ['mean', 'std', 'sem', 'count']
    ).reset_index()
    
    group_stats_path = os.path.join(output_dir, "group_id_summary_stats_by_condition.csv")
    group_stats.to_csv(group_stats_path, index=False)
    print(f"Group-level summary statistics saved to: {group_stats_path}")
    
    # --- VISUALIZATION ---
    print("\n--- Data successfully saved. Now proceeding to visualization. ---")
    
    print("\n--- Group-Level Mean Intrinsic Dimensionality (by Condition) ---")
    print(group_stats.to_string(index=False))

    print("\n--- Generating Group-Level Plot ---")
    try:
        g = sns.catplot(
            data=results_df,
            x='ROI',
            y='Intrinsic_Dim',
            col='Condition',
            kind='bar',
            col_wrap=2,
            order=roi_names,
            col_order=trial_types,
            errorbar='se',
            capsize=.05,
            palette='plasma',
            height=5,
            aspect=1.2
        )

        g.fig.suptitle('Group-Level Intrinsic Dimensionality by Condition (Voxel-Controlled)', y=1.03, fontsize=16)
        g.set_axis_labels("Region of Interest (ROI)", "Intrinsic Dimensionality (ID)")
        g.set_titles("Condition: {col_name}")
        g.despine(left=True)
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        plot_path = os.path.join(output_dir, "group_id_visualization_by_condition.png")
        plt.savefig(plot_path, dpi=300)
        print(f"Group plot saved to: {plot_path}")
    except Exception as e:
        print(f"\n--- VISUALIZATION FAILED ---")
        print(f"An error occurred during plotting: {e}")
        print("However, your computed results have been safely saved to the CSV files in the output directory.")

if __name__ == '__main__':
    main()
