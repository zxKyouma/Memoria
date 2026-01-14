import pandas as pd
from pathlib import Path
import shutil

# --- Configuration ---
# Set the top-level directory of your dataset.
# Must be the same as in the first script.
BASE_DIR = Path("/home/zlow/by84/Brandon/memoria")

# The name of the backup directory to create inside each 'func' folder.
BACKUP_DIR_NAME = "original_events"
# --- End of Configuration ---


def main():
    """
    Main function to finalize the event files.
    - Backs up the original _events.tsv files.
    - Renames the _events_corrected.tsv files to the standard BIDS name.
    """
    print("="*60)
    print("Starting BIDS event file finalization script.")
    print(f"Searching for corrected event files in: {BASE_DIR}")
    print("="*60)

    # Use glob to find all files that have been corrected
    search_pattern = "sub-*/ses-*/func/*_events_corrected.tsv"
    corrected_files = sorted(list(BASE_DIR.glob(search_pattern)))

    if not corrected_files:
        print("\n[INFO] No `_events_corrected.tsv` files found. Did you run the correction script first?")
        return

    print(f"\nFound {len(corrected_files)} corrected event files to finalize.\n")

    for corrected_path in corrected_files:
        # --- 1. Define all the paths we need ---
        # The directory where the file lives (e.g., .../func/)
        func_dir = corrected_path.parent
        # The path for the new backup directory
        backup_dir = func_dir / BACKUP_DIR_NAME
        
        # The standard BIDS filename for the original event file
        original_filename = corrected_path.name.replace('_corrected', '')
        original_path = func_dir / original_filename
        
        print(f"--- Processing: {corrected_path.name} ---")

        # --- 2. Check if the original file exists before trying to move it ---
        if not original_path.exists():
            print(f"  [WARNING] Original file not found at: {original_path}")
            print("  This might mean you already ran this script. Renaming corrected file anyway.")
        else:
            # --- 3. Create the backup directory if it doesn't exist ---
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # --- 4. Move the original file to the backup directory ---
            destination_path = backup_dir / original_filename
            print(f"  -> Backing up '{original_path.name}' to '{backup_dir.name}/'")
            shutil.move(src=original_path, dst=destination_path)

        # --- 5. Rename the corrected file to the original BIDS name ---
        print(f"  -> Renaming '{corrected_path.name}' to '{original_filename}'")
        corrected_path.rename(original_path)
        print("  Done.\n")
        
    print("="*60)
    print("Finalization complete. Your event files are ready for analysis.")
    print("Originals have been safely moved to `original_events` sub-directories.")
    print("="*60)


if __name__ == "__main__":
    main()
