import os
import pandas as pd
import re

def modify_retrieval_events_files(base_directory):
    """
    Iterates through subject directories to modify the trial_type in
    task-retrieval events.tsv files, handling special 'NA' cases.

    For each subject, this script locates the events.tsv file for the
    'retrieval' task. It then extracts the number from the 'item' column
    and inserts it into the 'trial_type' string.

    It handles two conditions:
    1.  Standard trial_type: 'faceRecog_unsure_correct' with item 'fok_09378.png'
        becomes 'faceRecog09378_unsure_correct'.
    2.  NA trial_type: 'retFeedbackNA' with item 'mob_00760.png'
        becomes 'retFeedback00760_NA'.

    Args:
        base_directory (str): The absolute path to the directory containing
                              the subject folders (e.g., os.environ.get("MEMORIA_ROOT", ".")).
    """
    print(f"Starting to process files in: {base_directory}")

    # Loop through each item in the base directory
    for subject_id in os.listdir(base_directory):
        if subject_id.startswith('sub-'):
            events_file_path = os.path.join(
                base_directory,
                subject_id,
                'ses-1',
                'func',
                f'{subject_id}_ses-1_task-consolidation_events.tsv'
            )

            # Check if the events.tsv file exists
            if os.path.isfile(events_file_path):
                try:
                    # Read the TSV file into a pandas DataFrame
                    events_df = pd.read_csv(events_file_path, sep='\t')

                    # Define a function to apply the modification
                    def update_trial_type(row):
                        trial_type = str(row['trial_type'])
                        item = str(row['item'])
                        
                        # Use regex to find one or more digits in the 'item' string
                        number_match = re.search(r'\d+', item)
                        if not number_match:
                            return trial_type # Return original if no number found in item

                        number = number_match.group(0)

                        # Check for the special 'NA' suffix case
                        if trial_type.endswith('NA'):
                            base_type = trial_type[:-2] # Get the part before 'NA'
                            return f"{base_type}{number}_NA"
                        
                        # Handle the standard case with an underscore
                        elif '_' in trial_type:
                            trial_type_parts = trial_type.split('_', 1)
                            return f"{trial_type_parts[0]}{number}_{trial_type_parts[1]}"
                        
                        # Fallback for any other unexpected format
                        else:
                            return trial_type

                    # Apply the function to the 'trial_type' column
                    # The 'axis=1' ensures the function is applied row-wise
                    events_df['trial_type'] = events_df.apply(update_trial_type, axis=1)

                    # Save the modified DataFrame back to the original file
                    events_df.to_csv(events_file_path, sep='\t', index=False)
                    print(f"Successfully modified: {events_file_path}")

                except Exception as e:
                    print(f"ERROR: Could not process {events_file_path}. Reason: {e}")
            else:
                print(f"INFO: File not found, skipping: {events_file_path}")

# --- SCRIPT EXECUTION ---

# IMPORTANT: Before running, ensure this path is correct and
# consider backing up your data.
base_dir = os.environ.get("MEMORIA_ROOT", ".")

modify_retrieval_events_files(base_dir)

print("\nProcessing complete.")
