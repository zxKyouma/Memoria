import pandas as pd
from pathlib import Path

# --- Configuration ---
# Set the top-level directory of your dataset.
# The script will search for event files within this directory.
BASE_DIR = Path("/home/zlow/by84/Brandon/memoria")

# Define the fixed sequence of event types for trials with missing responses.
# Using a "_noResp" suffix is good practice for clarity.
EVENT_SEQUENCE = ['Face_noResp', 'Enc_noResp', 'Recall_noResp', 'Feedback_noResp']

# --- End of Configuration ---

def correct_events_file(file_path: Path):
    """
    Loads an events.tsv file, finds trials with 'NA' trial_types,
    and fills them in based on the expected fixed sequence.

    Args:
        file_path (Path): The path to the BIDS _events.tsv file.
    """
    print(f"--- Processing: {file_path.name} ---")
    
    try:
        # Load the file. Treat both 'NA' and 'n/a' as missing values.
        df = pd.read_csv(file_path, sep='\t', na_values=['NA', 'n/a'])
    except Exception as e:
        print(f"  [ERROR] Could not read file: {e}")
        return

    # Check if 'trial_type' column exists and has missing values
    if 'trial_type' not in df.columns or df['trial_type'].isna().sum() == 0:
        print("  -> No missing trial types found. Skipping.")
        return

    # Find all unique items that have at least one NA in trial_type
    # Using .copy() to avoid SettingWithCopyWarning
    items_with_na = df.loc[df['trial_type'].isna(), 'item'].unique()

    if len(items_with_na) == 0:
        print("  -> No items associated with missing trial types. Skipping.")
        return
        
    print(f"  Found {len(items_with_na)} item(s) with missing responses: {', '.join(items_with_na)}")

    # Loop through each of these items and fill in the missing trial types
    for item_id in items_with_na:
        # Get the row indices for all events corresponding to this item
        item_indices = df[df['item'] == item_id].index
        
        # Sanity check: Ensure there are exactly 4 events for this item
        if len(item_indices) == len(EVENT_SEQUENCE):
            # Assign the fixed sequence to the 'trial_type' column for these rows
            df.loc[item_indices, 'trial_type'] = EVENT_SEQUENCE
            print(f"    - Corrected item '{item_id}' with sequence: {EVENT_SEQUENCE}")
        else:
            # This is an important warning for quality control!
            print(f"  [WARNING] Item '{item_id}' in file {file_path.name} has "
                  f"{len(item_indices)} events, not the expected {len(EVENT_SEQUENCE)}. "
                  "This trial was NOT corrected.")

    # --- Save the corrected file ---
    # Create a new filename to avoid overwriting the original.
    output_filename = file_path.stem + '_corrected' + file_path.suffix
    output_path = file_path.parent / output_filename
    
    # Save the modified dataframe to the new file
    # na_rep='n/a' is the BIDS standard for representing missing values.
    df.to_csv(output_path, sep='\t', index=False, na_rep='n/a')
    print(f"  -> Successfully saved corrected file to: {output_path}\n")


def main():
    """
    Main function to find and process all event files.
    """
    print("="*50)
    print("Starting BIDS event file correction script.")
    print(f"Searching in base directory: {BASE_DIR}")
    print("="*50)

    # Use glob to find all files matching the pattern recursively
    # This pattern looks for any _events.tsv file inside a /func/ directory
    # for any subject and session.
    search_pattern = "sub-*/ses-*/func/*_events.tsv"
    event_files = sorted(list(BASE_DIR.glob(search_pattern)))

    if not event_files:
        print("\n[ERROR] No `_events.tsv` files found matching the pattern.")
        print(f"Please check that the BASE_DIR is correct and your files follow BIDS format.")
        return

    print(f"\nFound {len(event_files)} event files to process.\n")

    # Process each file found
    for file in event_files:
        # We don't want to process files that have already been corrected
        if "_corrected" in file.name:
            continue
        correct_events_file(file)

    print("="*50)
    print("Automation complete.")
    print("="*50)


if __name__ == "__main__":
    main()
