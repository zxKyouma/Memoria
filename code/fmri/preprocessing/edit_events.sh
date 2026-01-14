#!/bin/bash

# The base directory where your subjects are located.
# The script will start searching from here.
BASE_DIR="/home/zlow/by84/Brandon/memoria"

# Check if the directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Base directory '$BASE_DIR' not found."
    exit 1
fi

echo "Starting in directory: $BASE_DIR"
echo "Searching for '*_task-forgetexp_events.tsv' files to process..."
echo "--------------------------------------------------------"

# Find all relevant events.tsv files and loop through them
# Using -print0 and read -d '' is the safest way to handle filenames with spaces or special characters
find "$BASE_DIR" -type f -name 'sub-*_task-forgetexp_events.tsv' -print0 | while IFS= read -r -d '' filepath; do
    echo "Processing file: $filepath"

    # Use awk to process the file and write to a temporary file.
    # The '&& mv' part ensures the original is only replaced if awk succeeds.
    awk '
    BEGIN {
        # Set Input and Output Field Separator to a tab character
        FS=OFS="\t"
    }
    # For the first line (the header), just print it and move to the next line
    NR == 1 {
        print
        next
    }
    # For all other lines, perform the modification
    {
        # Get the numbers from the item column ($4) by removing all non-digit characters
        item_num = $4
        gsub(/[^0-9]+/, "", item_num)

        # Find the position of the first underscore in the trial_type column ($3)
        split_pos = index($3, "_")

        # Reconstruct the new trial_type if an underscore exists
        if (split_pos > 0) {
            prefix = substr($3, 1, split_pos - 1)
            suffix = substr($3, split_pos)
            $3 = prefix item_num suffix
        }

        # Print the entire modified line. Awk rebuilds the line using the OFS.
        print $1, $2, $3, $4
    }
    ' "$filepath" > "$filepath.tmp" && mv "$filepath.tmp" "$filepath"

    echo "Done."
    echo "--------------------------------------------------------"

done

echo "All matching files have been updated."
