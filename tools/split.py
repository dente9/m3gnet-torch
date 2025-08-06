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
DATA_DIR = "./data/tep"
OUTPUT_DIR = "./data/split_medium"

# 2. Splitting Ratios
TEST_SPLIT_RATIO = 0.05
VAL_SPLIT_RATIO = 0.05

# 3. Reproducibility
RANDOM_STATE = 42

# 正则表达式，用于从文件名中提取基础ID (e.g., "mp-12345" from "mp-12345_relaxed.cif")
# 这个正则表达式现在很健壮，可以匹配 "mp-12345", "mvc-1234", "mp-12345.cif" 等
BASE_ID_PATTERN = re.compile(r"^(m[pvc]{1,2}-\d+)")

# ==============================================================================
# ---                       CORE IMPLEMENTATION                          ---
# ==============================================================================


def extract_base_id(filename: str):
    """从给定的文件名字符串中提取基础ID。"""
    if not isinstance(filename, str):
        return None
    match = BASE_ID_PATTERN.match(filename)
    return match.group(1) if match else None


def group_files_by_material(data_dir: str) -> dict:
    """
    扫描目录中的.cif文件，并根据基础材料ID进行分组。
    这是获取【真实、完整文件名】的唯一来源。
    """
    print("Step 1: Grouping CIF files by base material ID...")
    material_groups = defaultdict(list)

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Source data directory not found: {data_dir}")

    cif_filenames = [f for f in os.listdir(data_dir) if f.endswith(".cif")]

    if not cif_filenames:
        raise FileNotFoundError(f"No .cif files found in {data_dir}")

    skipped_count = 0
    for filename in tqdm(cif_filenames, desc="Scanning and Grouping files"):
        base_id = extract_base_id(filename)
        if base_id:
            # 关键修改：存储【完整】的文件名，而不是处理过的
            material_groups[base_id].append(filename)
        else:
            print(f"Warning: Could not determine base ID for file: {filename}")
            skipped_count += 1

    if skipped_count > 0:
        print(
            f"--> Warning: Skipped {skipped_count} files due to unmatched ID pattern."
        )
    print(
        f"-> Found {len(material_groups)} unique material groups from {len(cif_filenames)} files."
    )
    return dict(material_groups)


def create_splits(material_groups: dict) -> tuple:
    """将材料组拆分为训练、验证和测试集。"""
    print(
        "\nStep 2: Splitting material groups into train, validation, and test sets..."
    )
    base_ids = list(material_groups.keys())

    if len(base_ids) == 0:
        raise ValueError("No material groups found. Cannot create splits.")

    if TEST_SPLIT_RATIO + VAL_SPLIT_RATIO >= 1.0:
        raise ValueError(
            "Sum of TEST_SPLIT_RATIO and VAL_SPLIT_RATIO must be less than 1.0"
        )

    train_val_ids, test_ids = train_test_split(
        base_ids, test_size=TEST_SPLIT_RATIO, random_state=RANDOM_STATE
    )

    val_size_adjusted = (
        VAL_SPLIT_RATIO / (1.0 - TEST_SPLIT_RATIO)
        if (1.0 - TEST_SPLIT_RATIO) > 0
        else 0
    )

    if val_size_adjusted > 0 and len(train_val_ids) > 1:
        train_ids, val_ids = train_test_split(
            train_val_ids, test_size=val_size_adjusted, random_state=RANDOM_STATE
        )
    else:
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
    【完全重构的函数】
    处理单个数据集划分 (train, val, or test):
    1. 从`split_data`中获取所有真实、完整的文件名。
    2. 为这些文件匹配属性，生成一个临时的、准确的DataFrame。
    3. 根据这个准确的DataFrame，生成新的、干净的 `id_prop.csv`。
    4. 复制文件，此时文件名100%正确。
    """
    print(f"\nProcessing '{split_name}' split...")
    split_dest_dir = os.path.join(dest_dir, split_name)
    os.makedirs(split_dest_dir, exist_ok=True)

    # 1. 获取当前split中所有【完整】的文件名
    all_full_filenames_in_split = [
        filename for filenames in split_data.values() for filename in filenames
    ]

    if not all_full_filenames_in_split:
        print(f"  -> No files to process for '{split_name}' split. Skipping.")
        return 0

    # 2. 基于这些真实存在的文件名，创建一个新的DataFrame
    df_split = pd.DataFrame(all_full_filenames_in_split, columns=["filename"])

    # 3. 为这个新的DataFrame提取'base_id'，用于匹配属性
    df_split["base_id"] = df_split["filename"].apply(extract_base_id)

    # 4. 从原始属性DataFrame中，将属性值【合并】到我们新的DataFrame中
    #    我们只关心'base_id'和'property'列
    df_props_to_merge = df_properties[["base_id", "property"]].drop_duplicates()
    df_final_split = pd.merge(df_split, df_props_to_merge, on="base_id", how="left")

    # 检查是否有文件没能匹配到属性
    unmatched_count = df_final_split["property"].isnull().sum()
    if unmatched_count > 0:
        print(
            f"  - Warning: {unmatched_count} files in '{split_name}' split did not find a matching property in id_prop.csv. They will be excluded from the split's csv."
        )
        df_final_split.dropna(subset=["property"], inplace=True)

    # 5. 保存标准的、两列的 id_prop.csv。现在'filename'列是完整且正确的！
    output_csv_path = os.path.join(split_dest_dir, "id_prop.csv")
    df_final_split[["filename", "property"]].to_csv(
        output_csv_path, index=False, sep=","
    )
    print(
        f"  - Saved id_prop.csv for '{split_name}' with {len(df_final_split)} entries."
    )

    # 6. 复制文件。我们从`df_final_split`中迭代，确保只复制那些有属性且真实存在的文件。
    print(f"  - Copying {len(df_final_split)} .cif files for '{split_name}' split...")
    for filename_to_copy in tqdm(
        df_final_split["filename"], desc=f"Copying {split_name} files"
    ):
        src_path = os.path.join(src_dir, filename_to_copy)
        dest_path = os.path.join(split_dest_dir, filename_to_copy)

        # 尽管我们知道文件存在，这层检查仍然是好习惯
        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_path)
        else:
            # 在新逻辑下，这个警告理论上不应该再出现
            print(
                f"    - UNEXPECTED WARNING: Source file not found, skipping copy: {src_path}"
            )

    print(
        f"  -> Finished processing '{split_name}' split. Saved to '{split_dest_dir}/'"
    )
    return len(df_final_split)


def run_preparation():
    """主执行函数"""

    # --- 1. 稳健地加载和解析属性CSV ---
    prop_csv_path = os.path.join(DATA_DIR, "id_prop.csv")
    if not os.path.exists(prop_csv_path):
        raise FileNotFoundError(f"Error: 'id_prop.csv' not found in {DATA_DIR}")

    print(f"Loading and parsing properties from {prop_csv_path}...")
    try:
        # 这个解析逻辑现在很灵活，能处理多种空格/逗号分隔符
        df_properties = pd.read_csv(prop_csv_path, sep=r"[\s,]+", engine="python")
        df_properties.columns = ["filename", "property"]  # 假设总是这两列
        df_properties["property"] = pd.to_numeric(df_properties["property"])
        df_properties.dropna(inplace=True)

        print(f"-> Successfully parsed {len(df_properties)} records from id_prop.csv.")

        # 关键步骤: 为原始属性表创建'base_id'列，用于后续匹配
        # 这个逻辑能处理带或不带.cif后缀的文件名
        df_properties["base_id"] = df_properties["filename"].apply(extract_base_id)

        initial_count = len(df_properties)
        df_properties.dropna(subset=["base_id"], inplace=True)
        dropped_count = initial_count - len(df_properties)
        if dropped_count > 0:
            print(
                f"-> Warning: Dropped {dropped_count} records from id_prop.csv because a valid base_id could not be extracted."
            )
        print(
            f"-> Extracted base_id for {len(df_properties)} property records for matching."
        )

    except Exception as e:
        raise ValueError(
            f"Failed to parse CSV file '{prop_csv_path}'. Please check its format. Error: {e}"
        )

    # --- 2. 基于文件系统进行分组和拆分 ---
    material_groups = group_files_by_material(DATA_DIR)
    train_split, val_split, test_split = create_splits(material_groups)

    # --- 3. 处理拆分（创建CSV并复制CIF） ---
    print("\nStep 3: Processing splits (creating new CSVs and copying CIFs)...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_count = process_and_copy_split(
        "train", train_split, df_properties, DATA_DIR, OUTPUT_DIR
    )
    val_count = process_and_copy_split(
        "val", val_split, df_properties, DATA_DIR, OUTPUT_DIR
    )
    test_count = process_and_copy_split(
        "test", test_split, df_properties, DATA_DIR, OUTPUT_DIR
    )

    # --- 4. 保存最终摘要 ---
    summary_info = {
        "source_directory": os.path.abspath(DATA_DIR),
        "output_directory": os.path.abspath(OUTPUT_DIR),
        "split_ratios": {
            "train": 1.0 - TEST_SPLIT_RATIO - VAL_SPLIT_RATIO,
            "validation": VAL_SPLIT_RATIO,
            "test": TEST_SPLIT_RATIO,
        },
        "total_material_groups_found": len(material_groups),
        "splits_info": {
            "train": {
                "num_groups": len(train_split),
                "num_structures_processed": train_count,
            },
            "validation": {
                "num_groups": len(val_split),
                "num_structures_processed": val_count,
            },
            "test": {
                "num_groups": len(test_split),
                "num_structures_processed": test_count,
            },
        },
    }
    output_json_path = os.path.join(OUTPUT_DIR, "dataset_split_summary.json")
    with open(output_json_path, "w") as f:
        json.dump(summary_info, f, indent=4)
    print(f"\n-> Saved final split summary to {output_json_path}")

    print("\nDataset preparation complete!")


if __name__ == "__main__":
    run_preparation()
