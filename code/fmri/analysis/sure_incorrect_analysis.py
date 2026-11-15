import os
import glob
import pandas as pd
import numpy as np
from nilearn.image import load_img, index_img
from nilearn.maskers import NiftiMasker
import skdim
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import pingouin as pg
import re

# --- USER CONFIGURATION ---
BIDS_DIR = os.environ.get("MEMORIA_ROOT", ".")
BIDS_TASK_NAME = "forgetexp"
ROIS_TO_ANALYZE = ["V1", "hV4", "LOIT", "phc", "hippocampus"]
RANDOM_SEED = 42
OUTPUT_DIR = "./id_analysis_sure_vs_incorrect"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# The main task phase whose trials we will split by behavior
BEHAVIORAL_TASK_PHASE = "Recall" 
# Minimum number of trials required for a condition to be included
MIN_TRIALS_FOR_ANALYSIS = 4 # Needs to be > a few for stable ID. Adjust if needed.
# --- END OF CONFIGURATION ---

def find_subjects(nibetaseries_dir):
    subject_dirs = glob.glob(os.path.join(nibetaseries_dir, "sub-*"))
    return sorted([os.path.basename(d).split('-')[1] for d in subject_dirs if os.path.isdir(d)])

def get_behavioral_info_for_betas(subject_id, session_id="1"):
    """Creates a master dataframe linking each beta image index to its behavioral outcome."""
    try:
        nibetaseries_events_template = os.path.join(
            BIDS_DIR, "derivatives", "nibetaseries_nosm", "nibetaseries",
            f"sub-{subject_id}", f"ses-{session_id}", "func",
            f"sub-{subject_id}_ses-{session_id}_task-{BIDS_TASK_NAME}_acq-*_space-T1w_desc-{BEHAVIORAL_TASK_PHASE}_events.tsv"
        )
        nibetaseries_events_file = glob.glob(nibetaseries_events_template)[0]
        nibetaseries_df = pd.read_csv(nibetaseries_events_file, sep='\t')
        nibetaseries_df['beta_index'] = nibetaseries_df.index

        bids_events_file = os.path.join(BIDS_DIR, f"sub-{subject_id}", f"ses-{session_id}", "func", f"sub-{subject_id}_ses-{session_id}_task-{BIDS_TASK_NAME}_events.tsv")
        bids_df = pd.read_csv(bids_events_file, sep='\t')
        
        bids_df = bids_df.sort_values('onset'); nibetaseries_df = nibetaseries_df.sort_values('onset')
        merged_df = pd.merge_asof(nibetaseries_df, bids_df, on='onset', direction='nearest', suffixes=('_nibs', '_bids'))

        pattern = re.compile(r'(\w+?)(\d+)_(\w+?)(?:_(\w+))?$')
        parsed_data = merged_df['trial_type_bids'].str.extract(pattern)
        parsed_data.columns = ['Task_Phase_bids', 'ItemID', 'Confidence', 'Accuracy']
        
        final_df = pd.concat([merged_df, parsed_data], axis=1)
        final_df = final_df[final_df['Task_Phase_bids'] == BEHAVIORAL_TASK_PHASE].copy()
        
        # Create a single label for our conditions of interest
        final_df['Analysis_Condition'] = final_df['Confidence'] + "_" + final_df['Accuracy']
        
        return final_df.dropna(subset=['beta_index', 'Analysis_Condition'])
    except Exception as e:
        print(f"Warning: Could not process events for sub-{subject_id}. Error: {e}")
        return pd.DataFrame()


def calculate_id_for_subset(beta_series_img, beta_indices, roi_mask_path):
    """Calculates ID on a specific subset of beta images defined by their indices."""
    np.random.seed(RANDOM_SEED)
    
    # Select only the beta images for this condition
    condition_betas_img = index_img(beta_series_img, beta_indices)
    
    masker = NiftiMasker(mask_img=roi_mask_path, standardize=False)
    voxel_data = masker.fit_transform(condition_betas_img)
    
    # Voxel subsampling is not needed here as we compare within-subject, within-ROI
    id_estimator = skdim.id.TwoNN().fit(voxel_data)
    return id_estimator.dimension_


def main():
    """
    Main function to compare ID for "Sure-Correct" vs. "Sure-Incorrect" trials.
    """
    print('--- Starting "Sure-Correct" vs. "Sure-Incorrect" ID Analysis ---')
    nibetaseries_base_dir = os.path.join(BIDS_DIR, "derivatives", "nibetaseries_nosm", "nibetaseries")
    all_subjects = find_subjects(nibetaseries_base_dir)

    analysis_results = []
    subjects_included = []

    for subject_id in tqdm(all_subjects, desc="Processing Subjects"):
        # Get the behavioral data linked to beta indices
        behavioral_df = get_behavioral_info_for_betas(subject_id)
        if behavioral_df.empty: continue

        # Identify indices for our two conditions of interest
        sure_correct_indices = behavioral_df[behavioral_df['Analysis_Condition'] == 'sure_correct']['beta_index'].astype(int).tolist()
        sure_incorrect_indices = behavioral_df[behavioral_df['Analysis_Condition'] == 'sure_incorrect']['beta_index'].astype(int).tolist()

        # --- CRITICAL STEP: Check if subject has enough trials for BOTH conditions ---
        if len(sure_correct_indices) < MIN_TRIALS_FOR_ANALYSIS or len(sure_incorrect_indices) < MIN_TRIALS_FOR_ANALYSIS:
            continue
        
        # If they pass the check, load the full beta-series file once
        try:
            beta_series_template = os.path.join(nibetaseries_base_dir, f"sub-{subject_id}", "ses-1", "func", f"*_desc-{BEHAVIORAL_TASK_PHASE}_betaseries.nii.gz")
            beta_series_file_path = glob.glob(beta_series_template)[0]
            beta_series_img = load_img(beta_series_file_path)
        except IndexError:
            continue

        # This subject is good to go, add them to our list
        subjects_included.append(subject_id)

        # Now, calculate ID for both conditions across all ROIs
        for roi_name in ROIS_TO_ANALYZE:
            roi_mask_path = os.path.join(BIDS_DIR, "derivatives", "roi_masks", f"sub-{subject_id}", "anat", f"sub-{subject_id}_space-T1w_desc-{roi_name}_resampled-betaseries_mask.nii.gz")
            if not os.path.exists(roi_mask_path): continue

            try:
                # Calculate ID for Sure-Correct trials
                id_sc = calculate_id_for_subset(beta_series_img, sure_correct_indices, roi_mask_path)
                analysis_results.append({"Subject": subject_id, "ROI": roi_name, "Condition": "Sure_Correct", "Intrinsic_Dim": id_sc})
                
                # Calculate ID for Sure-Incorrect trials
                id_si = calculate_id_for_subset(beta_series_img, sure_incorrect_indices, roi_mask_path)
                analysis_results.append({"Subject": subject_id, "ROI": roi_name, "Condition": "Sure_Incorrect", "Intrinsic_Dim": id_si})
            except Exception as e:
                print(f"Error during ID calculation for sub-{subject_id}, ROI-{roi_name}: {e}")

    if not analysis_results:
        print("\nAnalysis failed. No subjects had sufficient 'Sure-Correct' AND 'Sure-Incorrect' trials.")
        return

    # --- Statistical Analysis and Visualization ---
    results_df = pd.DataFrame(analysis_results)
    print(f"\n--- Analysis complete. {len(subjects_included)} subjects had enough trials to be included. ---")
    
    # Save the detailed results
    results_df.to_csv(os.path.join(OUTPUT_DIR, "sure_correct_vs_incorrect_id_results.csv"), index=False)

    print("\n--- Paired T-test Results (Sure_Correct vs. Sure_Incorrect) ---")
    # Run a paired t-test for each ROI
    stats = results_df.groupby('ROI').apply(
        lambda df: pg.ttest(df[df['Condition'] == 'Sure_Correct']['Intrinsic_Dim'], 
                              df[df['Condition'] == 'Sure_Incorrect']['Intrinsic_Dim'], paired=True)
    ).reset_index()
    print(stats[['ROI', 'T', 'dof', 'alternative', 'p-val', 'CI95%', 'cohen-d', 'BF10']])
    stats.to_csv(os.path.join(OUTPUT_DIR, "stats_sure_correct_vs_incorrect.csv"), index=False)
    
    # Visualization
    plt.figure(figsize=(12, 7))
    sns.catplot(
        data=results_df,
        x='ROI',
        y='Intrinsic_Dim',
        hue='Condition',
        kind='point',
        order=ROIS_TO_ANALYZE,
        dodge=True,
        errorbar='se',
        capsize=0.05,
        height=6,
        aspect=1.5
    )
    plt.title(f'ID for Confident Trials by Accuracy ({len(subjects_included)} Subjects)')
    plt.ylabel("Intrinsic Dimensionality (ID)")
    plt.xlabel("Region of Interest (ROI)")
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "plot_sure_correct_vs_incorrect.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\nPlot saved to {plot_path}")

if __name__ == '__main__':
    main()
