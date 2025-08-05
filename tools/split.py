# prepare_dataset_v4.py

import os
import re
import json
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import defaultdict

# ==============================================================================
# ---                           CONFIGURATION                            ---
# ==============================================================================

# 1. Paths
# Path to the folder containing ALL .cif files and the main id_prop.csv
DATA_DIR = "./data_all"
# Directory where the new train/val/test subdirectories will be created
OUTPUT_DIR = "./processed_data"

# 2. Splitting Ratios
TEST_SPLIT_RATIO = 0.05
VAL_SPLIT_RATIO = 0.05

# 3. Reproducibility
RANDOM_STATE = 42

# ==============================================================================
# ---                       CORE IMPLEMENTATION                          ---
# ==============================================================================


def group_files_by_material(data_dir: str) -> dict:
    """Scans a directory for .cif files and groups them by base material ID."""
    print("Step 1: Grouping CIF files by base material ID...")
    base_id_pattern = re.compile(r"^(m[pvc]{1,2}-\d+)")
    material_groups = defaultdict(list)
    cif_filenames = [f for f in os.listdir(data_dir) if f.endswith(".cif")]

    skipped_count = 0
    for filename in tqdm(cif_filenames, desc="Grouping files"):
        basename = os.path.splitext(filename)[0]
        match = base_id_pattern.match(basename)
        if match:
            base_id = match.group(1)
            material_groups[base_id].append(basename)
        else:
            skipped_count += 1

    if skipped_count > 0:
        print(f"Warning: Could not determine base ID for {skipped_count} files.")
    print(f"-> Found {len(material_groups)} unique material groups.")
    return dict(material_groups)


def create_splits(material_groups: dict) -> tuple:
    """Splits material groups into training, validation, and test sets."""
    print(
        "\nStep 2: Splitting material groups into train, validation, and test sets..."
    )
    base_ids = list(material_groups.keys())

    train_val_ids, test_ids = train_test_split(
        base_ids, test_size=TEST_SPLIT_RATIO, random_state=RANDOM_STATE
    )
    val_size_adjusted = VAL_SPLIT_RATIO / (1.0 - TEST_SPLIT_RATIO)
    train_ids, val_ids = train_test_split(
        train_val_ids, test_size=val_size_adjusted, random_state=RANDOM_STATE
    )

    train_split = {id: material_groups[id] for id in train_ids}
    val_split = {id: material_groups[id] for id in val_ids}
    test_split = {id: material_groups[id] for id in test_ids}

    print(f"-> Splitting complete:")
    print(f"  - Training groups:   {len(train_split)}")
    print(f"  - Validation groups: {len(val_split)}")
    print(f"  - Test groups:       {len(test_split)}")
    return train_split, val_split, test_split


def process_and_copy_split(
    split_name: str,
    split_data: dict,
    df_properties: pd.DataFrame,
    id_col: str,
    src_dir: str,
    dest_dir: str,
):
    """
    Processes a single split (train, val, or test):
    1. Creates the destination subdirectory.
    2. Filters the main DataFrame to create the split's id_prop.csv.
    3. Copies the corresponding .cif files.
    """
    split_dest_dir = os.path.join(dest_dir, split_name)
    os.makedirs(split_dest_dir, exist_ok=True)

    all_basenames_in_split = {
        basename
        for basenames_list in split_data.values()
        for basename in basenames_list
    }

    # Filter the DataFrame for the current split
    df_split = df_properties[
        df_properties[id_col].astype(str).isin(all_basenames_in_split)
    ].copy()

    # Save the new id_prop.csv
    output_csv_path = os.path.join(split_dest_dir, "id_prop.csv")
    df_split.to_csv(output_csv_path, index=False)

    # <<<<<<<<<<<<<<<<< 新增的文件复制逻辑 <<<<<<<<<<<<<<<<<
    print(f"  - Copying {len(df_split)} .cif files for '{split_name}' split...")
    for basename in tqdm(df_split[id_col], desc=f"Copying {split_name} files"):
        cif_filename = f"{basename}.cif"
        src_path = os.path.join(src_dir, cif_filename)
        dest_path = os.path.join(split_dest_dir, cif_filename)

        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_path)  # copy2 preserves metadata
        else:
            print(f"    - Warning: Source file not found, skipping copy: {src_path}")
    # <<<<<<<<<<<<<<<<< 逻辑结束 <<<<<<<<<<<<<<<<<

    print(
        f"  -> Finished processing '{split_name}' split. Saved to '{split_dest_dir}/'"
    )
    return len(df_split)


def run_preparation():
    """Main execution function."""

    # --- 1. Load Properties CSV and Identify Columns ---
    prop_csv_path = os.path.join(DATA_DIR, "id_prop.csv")
    if not os.path.exists(prop_csv_path):
        raise FileNotFoundError(f"Error: 'id_prop.csv' not found in {DATA_DIR}")

    print(f"Loading properties from {prop_csv_path}...")
    df_properties = pd.read_csv(prop_csv_path)

    if len(df_properties.columns) < 2:
        raise ValueError(
            "id_prop.csv must have at least two columns (ID and property)."
        )

    id_column_name = df_properties.columns[0]
    property_column_name = df_properties.columns[1]
    print(
        f"-> Automatically detected columns: ID='{id_column_name}', Property='{property_column_name}'"
    )

    # --- 2. Group, Split, and Process Files ---
    material_groups = group_files_by_material(DATA_DIR)
    train_split, val_split, test_split = create_splits(material_groups)

    print("\nStep 3: Processing splits (creating CSVs and copying CIFs)...")

    train_info = process_and_copy_split(
        "train", train_split, df_properties, id_column_name, DATA_DIR, OUTPUT_DIR
    )
    val_info = process_and_copy_split(
        "val", val_split, df_properties, id_column_name, DATA_DIR, OUTPUT_DIR
    )
    test_info = process_and_copy_split(
        "test", test_split, df_properties, id_column_name, DATA_DIR, OUTPUT_DIR
    )

    # --- 4. Save Final Summary ---
    summary_info = {
        "source_directory": os.path.abspath(DATA_DIR),
        "output_directory": os.path.abspath(OUTPUT_DIR),
        "split_ratios": {
            "train": 1 - TEST_SPLIT_RATIO - VAL_SPLIT_RATIO,
            "validation": VAL_SPLIT_RATIO,
            "test": TEST_SPLIT_RATIO,
        },
        "total_material_groups": len(material_groups),
        "splits": {
            "train": {"num_groups": len(train_split), "num_structures": train_info},
            "validation": {"num_groups": len(val_split), "num_structures": val_info},
            "test": {"num_groups": len(test_split), "num_structures": test_info},
        },
    }
    output_json_path = os.path.join(OUTPUT_DIR, "dataset_split_summary.json")
    with open(output_json_path, "w") as f:
        json.dump(summary_info, f, indent=4)
    print(f"\n-> Saved final split summary to {output_json_path}")

    print("\nDataset preparation complete!")


if __name__ == "__main__":
    run_preparation()
