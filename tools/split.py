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
DATA_DIR = "./data/bbb"
OUTPUT_DIR = "./data/split_small"

# 2. Splitting Ratios
TEST_SPLIT_RATIO = 0.05
VAL_SPLIT_RATIO = 0.05

# 3. Reproducibility
RANDOM_STATE = 42

# 正则表达式，用于从文件名中提取基础ID (e.g., "mp-12345" from "mp-12345_relaxed.cif")
BASE_ID_PATTERN = re.compile(r"^(m[pvc]{1,2}-\d+)")

# ==============================================================================
# ---                       CORE IMPLEMENTATION                          ---
# ==============================================================================


def group_files_by_material(data_dir: str) -> dict:
    """Scans a directory for .cif files and groups them by base material ID."""
    print("Step 1: Grouping CIF files by base material ID...")
    material_groups = defaultdict(list)

    # 只处理.cif文件
    cif_filenames = [f for f in os.listdir(data_dir) if f.endswith(".cif")]

    skipped_count = 0
    for filename in tqdm(cif_filenames, desc="Grouping files"):
        # 从完整文件名中匹配基础ID
        match = BASE_ID_PATTERN.match(filename)
        if match:
            base_id = match.group(1)
            # 存储不带后缀的文件名
            material_groups[base_id].append(os.path.splitext(filename)[0])
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

    if len(base_ids) == 0:
        raise ValueError(
            "No material groups found. Cannot create splits. Check your DATA_DIR."
        )

    # 确保拆分比例合理
    if TEST_SPLIT_RATIO + VAL_SPLIT_RATIO >= 1.0:
        raise ValueError(
            "Sum of TEST_SPLIT_RATIO and VAL_SPLIT_RATIO must be less than 1.0"
        )

    train_val_ids, test_ids = train_test_split(
        base_ids, test_size=TEST_SPLIT_RATIO, random_state=RANDOM_STATE
    )
    # 调整验证集在剩余数据中的比例
    val_size_adjusted = (
        VAL_SPLIT_RATIO / (1.0 - TEST_SPLIT_RATIO)
        if (1.0 - TEST_SPLIT_RATIO) > 0
        else 0
    )

    if val_size_adjusted > 0 and len(train_val_ids) > 1:
        train_ids, val_ids = train_test_split(
            train_val_ids, test_size=val_size_adjusted, random_state=RANDOM_STATE
        )
    else:  # 如果剩余数据不够拆分，全部分给训练集
        train_ids = train_val_ids
        val_ids = []

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
    src_dir: str,
    dest_dir: str,
):
    """
    Processes a single split (train, val, or test):
    1. Creates the destination subdirectory.
    2. Filters the main DataFrame using the 'base_id' column.
    3. Saves the standard, two-column id_prop.csv.
    4. Copies the corresponding .cif files.
    """
    split_dest_dir = os.path.join(dest_dir, split_name)
    os.makedirs(split_dest_dir, exist_ok=True)

    # 获取当前split中所有的基础ID
    all_base_ids_in_split = set(split_data.keys())

    # <<<<<<<<<<<<<<<<< 核心匹配逻辑 <<<<<<<<<<<<<<<<<
    # 使用我们之前创建的 'base_id' 列来进行过滤，确保100%正确匹配
    df_split = df_properties[
        df_properties["base_id"].isin(all_base_ids_in_split)
    ].copy()

    # 准备写入标准的CSV，只保留'filename'和'property'两列
    df_to_save = df_split[["filename", "property"]]

    # 保存为标准的、逗号分隔的CSV文件
    output_csv_path = os.path.join(split_dest_dir, "id_prop.csv")
    df_to_save.to_csv(output_csv_path, index=False, sep=",")

    print(f"  - Copying {len(df_split)} .cif files for '{split_name}' split...")
    # 从过滤后的DataFrame中获取完整文件名列表进行复制
    for cif_filename in tqdm(df_split["filename"], desc=f"Copying {split_name} files"):
        src_path = os.path.join(src_dir, cif_filename)
        dest_path = os.path.join(split_dest_dir, cif_filename)

        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_path)
        else:
            print(f"    - Warning: Source file not found, skipping copy: {src_path}")
    # <<<<<<<<<<<<<<<<< 逻辑结束 <<<<<<<<<<<<<<<<<

    print(
        f"  -> Finished processing '{split_name}' split. Saved to '{split_dest_dir}/'"
    )
    return len(df_split)


def run_preparation():
    """Main execution function."""

    # --- 1. Robustly Load and Parse Properties CSV ---
    prop_csv_path = os.path.join(DATA_DIR, "id_prop.csv")
    if not os.path.exists(prop_csv_path):
        raise FileNotFoundError(f"Error: 'id_prop.csv' not found in {DATA_DIR}")

    print(f"Loading and parsing non-standard properties from {prop_csv_path}...")

    try:
        # 读取整个文件为一列，不识别标题
        temp_df = pd.read_csv(
            prop_csv_path, header=None, sep=r"\s{1,}", engine="python"
        )
        if temp_df.shape[1] == 1:  # 如果还是只有一列（没有空格分隔）
            temp_df = temp_df[0].str.rsplit("-", n=1, expand=True)

        # 提取标题和数据
        header = (
            temp_df.iloc[0, 0].replace("property", ""),
            "property",
        )  # ('filename', 'property')
        data = temp_df.iloc[1:]

        # 创建一个干净的DataFrame
        df_properties = pd.DataFrame(data.values, columns=header)
        df_properties.rename(
            columns={df_properties.columns[0]: "filename"}, inplace=True
        )
        df_properties["property"] = pd.to_numeric(df_properties["property"])
        df_properties.dropna(inplace=True)

        print(f"-> Successfully parsed {len(df_properties)} records.")

        # <<<<<<<<<<<<<<<<< 关键新增步骤: 创建 'base_id' 列用于匹配 <<<<<<<<<<<<<<<<<
        def extract_base_id(filename):
            match = BASE_ID_PATTERN.match(str(filename))
            return match.group(1) if match else None

        df_properties["base_id"] = df_properties["filename"].apply(extract_base_id)
        # 丢掉那些无法提取base_id的行
        df_properties.dropna(subset=["base_id"], inplace=True)
        print(f"-> Extracted base_id for {len(df_properties)} records for matching.")

    except Exception as e:
        raise ValueError(f"Failed to parse CSV file: {e}")

    # --- 2. Group, Split, and Process Files ---
    material_groups = group_files_by_material(DATA_DIR)
    train_split, val_split, test_split = create_splits(material_groups)

    print("\nStep 3: Processing splits (creating CSVs and copying CIFs)...")

    train_info = process_and_copy_split(
        "train", train_split, df_properties, DATA_DIR, OUTPUT_DIR
    )
    val_info = process_and_copy_split(
        "val", val_split, df_properties, DATA_DIR, OUTPUT_DIR
    )
    test_info = process_and_copy_split(
        "test", test_split, df_properties, DATA_DIR, OUTPUT_DIR
    )

    # --- 4. Save Final Summary ---
    summary_info = {
        "source_directory": os.path.abspath(DATA_DIR),
        "output_directory": os.path.abspath(OUTPUT_DIR),
        "split_ratios": {
            "train": 1.0 - TEST_SPLIT_RATIO - VAL_SPLIT_RATIO,
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
