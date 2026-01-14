# File: compute_rsa_debug.py (run this new version)

import os
import glob
import re
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, ttest_1samp
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import reduce

# --- PyTorch & Model Imports ---
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from model import BetaVAE

# ==============================================================================
#                             CONFIGURATION (Unchanged)
# ==============================================================================
BIDS_DIR = "/home/zlow/by84/Brandon/memoria"
SESSION_LABEL = "1"
TASK_LABEL = "Enc"
MODEL_PATH = "./outputs/saved_models/bvae_beta2_latent32_final.pth"
LATENT_DIM = 32
IMAGE_DATA_PATH = './data'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAE_LAYERS_TO_TEST = ['enc_layer_1', 'enc_layer_2', 'enc_layer_3', 'enc_layer_4', 'latent_z']
ROI_NAMES_IN_ORDER = ["V1", "hV4", "LOIT", "Fusiform", "PHC", "HPC", "Amygdala"]
ROI_FILENAME_MAP = {"V1": "V1", "hV4": "hV4", "LOIT": "LOIT", "PHC": "phc", "HPC": "hippocampus", "Amygdala": "amygdala", "Fusiform": "FUS"}
OUTPUT_DIR = "./rsa_results_revised"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
#           HELPER FUNCTIONS (Unchanged)
# ==============================================================================
def find_subjects(nibetaseries_dir):
    subject_dirs = glob.glob(os.path.join(nibetaseries_dir, "sub-*"))
    return sorted([os.path.basename(d).split('-')[1] for d in subject_dirs if os.path.isdir(d)])

def compute_rdm(data_matrix):
    if data_matrix.shape[0] < 2: return np.array([]) # <--- NEW: Handle cases with < 2 items
    return squareform(pdist(data_matrix, metric='correlation'))

def extract_vae_activations():
    # ... (This function is unchanged) ...
    print(f"\n--- Extracting VAE activations from model: {MODEL_PATH} ---")
    model = BetaVAE(latent_dim=LATENT_DIM).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = datasets.ImageFolder(root=IMAGE_DATA_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    full_image_id_order = [re.search(r'_(\d{5})\.png', os.path.basename(p)).group(1) for p, _ in dataset.samples]
    layer_activations = {layer: [] for layer in VAE_LAYERS_TO_TEST}
    with torch.no_grad():
        for data, _ in dataloader:
            _, _, _, activations_dict = model(data.to(DEVICE))
            for layer_name in VAE_LAYERS_TO_TEST:
                act = activations_dict[layer_name].view(activations_dict[layer_name].size(0), -1).cpu().numpy()
                layer_activations[layer_name].append(act)
    for layer_name, act_list in layer_activations.items():
        layer_activations[layer_name] = np.concatenate(act_list, axis=0)
    return layer_activations, full_image_id_order

# ==============================================================================
#                       MAIN ANALYSIS PIPELINE
# ==============================================================================
def main():
    nibetaseries_base_dir = os.path.join(BIDS_DIR, "derivatives", "nibetaseries_nosm", "nibetaseries")
    all_subjects = find_subjects(nibetaseries_base_dir)
    vae_activations, full_image_id_order = extract_vae_activations()

    print("\n--- Aligning image sets across all subjects ---")
    subject_image_sets = []
    valid_subjects_for_intersection = [] # <--- NEW: Keep track of which subjects contribute
    for subject_id in tqdm(all_subjects, desc="Finding images per subject"):
        subject_func_path = os.path.join(nibetaseries_base_dir, f"sub-{subject_id}", f"ses-{SESSION_LABEL}", "func")
        beta_files = glob.glob(os.path.join(subject_func_path, f'*desc-{TASK_LABEL}*_betaseries.nii.gz'))
        available_ids = {re.search(r'desc-.*(\d{5})_betaseries', f).group(1) for f in beta_files if re.search(r'desc-.*(\d{5})_betaseries', f)}
        if available_ids:
            subject_image_sets.append(available_ids)
            valid_subjects_for_intersection.append(subject_id)

    if not subject_image_sets:
        print("CRITICAL ERROR: No subjects had any valid beta files. Aborting.")
        return
        
    common_image_ids = sorted(list(reduce(lambda s1, s2: s1.intersection(s2), subject_image_sets)))
    
    # --- CRITICAL DEBUGGING STEP ---
    print(f"\n\n--- DEBUGGING INFO ---") # <--- NEW
    print(f"Number of subjects contributing to intersection: {len(valid_subjects_for_intersection)}") # <--- NEW
    print(f"Number of common images found across ALL subjects: {len(common_image_ids)}") # <--- NEW
    if not common_image_ids: # <--- NEW
        print("FATAL ERROR: The intersection is empty. There are no images common to all subjects.") # <--- NEW
        print("This is the likely cause of the error. The script will now abort.") # <--- NEW
        return # <--- NEW
    if len(common_image_ids) < 2: # <--- NEW
        print(f"FATAL ERROR: Only {len(common_image_ids)} image(s) common to all subjects. Cannot compute RDM. Aborting.") # <--- NEW
        return # <--- NEW
    print(f"--- END DEBUGGING ---\n") # <--- NEW

    common_indices = [full_image_id_order.index(img_id) for img_id in common_image_ids]
    filtered_activations = {layer: act[common_indices, :] for layer, act in vae_activations.items()}
    model_rdms = {layer: compute_rdm(filtered_activations[layer]) for layer in VAE_LAYERS_TO_TEST}
    model_rdm_vectors = {layer: rdm[np.triu_indices(len(rdm), k=1)] for layer, rdm in model_rdms.items()}
    
    all_results = []
    # --- LOOP OVER ONLY THE VALID SUBJECTS ---
    for subject_id in tqdm(valid_subjects_for_intersection, desc="Processing Subjects for RSA"): # <--- MODIFIED
        subject_betas = {}
        subject_func_path = os.path.join(nibetaseries_base_dir, f"sub-{subject_id}", f"ses-{SESSION_LABEL}", "func")
        beta_files = glob.glob(os.path.join(subject_func_path, f'*desc-{TASK_LABEL}*_betaseries.nii.gz'))
        for f_path in beta_files:
            match = re.search(r'desc-.*(\d{5})_betaseries', f_path)
            if match and match.group(1) in common_image_ids:
                img_id = match.group(1)
                if img_id not in subject_betas: subject_betas[img_id] = []
                subject_betas[img_id].append(nib.load(f_path).get_fdata())

        ordered_avg_betas = [np.mean(np.stack(subject_betas[img_id], axis=-1), axis=-1) for img_id in common_image_ids]
        all_images_data = np.stack(ordered_avg_betas, axis=-1)

        for roi_label in ROI_NAMES_IN_ORDER:
            # ... (rest of the loop is the same, no need to change) ...
            roi_fname = ROI_FILENAME_MAP[roi_label]
            mask_path = os.path.join(BIDS_DIR, "derivatives", "roi_masks", f"sub-{subject_id}", "anat", f"sub-{subject_id}_ses-{SESSION_LABEL}_space-T1w_desc-{roi_fname}_resampled-betaseries_mask.nii.gz")
            if not os.path.exists(mask_path): continue
            mask_data = nib.load(mask_path).get_fdata().astype(bool)
            roi_beta_data = all_images_data[mask_data, :].T
            if roi_beta_data.shape[1] < 10: continue
            neural_rdm = compute_rdm(roi_beta_data)
            if neural_rdm.size == 0: continue # <--- NEW: Safety check
            neural_rdm_vec = neural_rdm[np.triu_indices(len(neural_rdm), k=1)]
            for layer_name in VAE_LAYERS_TO_TEST:
                rsa_corr, _ = spearmanr(model_rdm_vectors[layer_name], neural_rdm_vec)
                z_corr = np.arctanh(rsa_corr)
                all_results.append({'subject': subject_id, 'roi': roi_label, 'layer': layer_name, 'rsa_corr': rsa_corr, 'z_corr': z_corr})

    # ... (Aggregation and plotting code is unchanged) ...
    if not all_results:
        print("\nAnalysis finished but no results were generated. Please check the logs for errors or empty data sets.") # <--- MODIFIED
        return
    results_df = pd.DataFrame(all_results).dropna()
    results_df.to_csv(os.path.join(OUTPUT_DIR, "rsa_full_results_revised.csv"), index=False)
    group_stats = results_df.groupby(['roi', 'layer'])['rsa_corr'].agg(['mean', 'sem']).reset_index()
    print("\n\n" + "="*80)
    print("        HIERARCHICAL RSA: GROUP-LEVEL SIGNIFICANCE vs. ZERO (on z-transformed data)")
    print("="*80)
    p_values = []
    for index, row in group_stats.iterrows():
        correlations = results_df[(results_df['roi'] == row['roi']) & (results_df['layer'] == row['layer'])]['z_corr'].values
        t_stat, p_val = ttest_1samp(correlations, 0, alternative='greater')
        p_values.append(p_val)
    group_stats['p_value'] = p_values
    print(group_stats.to_string())
    print("="*80)
    heatmap_data = group_stats.pivot(index='roi', columns='layer', values='mean')
    heatmap_data = heatmap_data.reindex(index=ROI_NAMES_IN_ORDER, columns=VAE_LAYERS_TO_TEST)
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="viridis", linewidths=.5)
    plt.title(f'Hierarchical RSA: VAE Layers vs. Brain ROIs (ses-{SESSION_LABEL}, task-{TASK_LABEL})', fontsize=16, pad=20)
    plt.xlabel('VAE Model Layer', fontsize=12)
    plt.ylabel('Brain Region of Interest (ROI)', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.savefig(os.path.join(OUTPUT_DIR, "rsa_summary_heatmap_revised.png"), dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()
