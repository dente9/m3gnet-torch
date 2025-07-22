# make_small_dataset.py
import os
import shutil
import random
import pandas as pd

SOURCE_DIR = r"data\aaa"
TARGET_DIR = r"data\bbb"
N_PICK = 2000
SEED = 42

random.seed(SEED)

csv_path = os.path.join(SOURCE_DIR, "id_prop.csv")

# 1. 先尝试用逗号分隔，若失败再用制表符
try:
    df = pd.read_csv(csv_path, sep=',')
except pd.errors.ParserError:
    df = pd.read_csv(csv_path, sep='\t')

# 2. 如果只有一列，说明整个行被当成一个字符串，再手动拆分
if df.shape[1] == 1:
    col = df.columns[0]
    # 按逗号拆分
    df = df[col].str.split(',', expand=True)
    df.columns = ['filename', 'property']
    # 把 property 转成 float
    df['property'] = pd.to_numeric(df['property'], errors='coerce')

# 3. 打印确认
print("列名：", df.columns.tolist())
print(df.head())

# 4. 随机抽取
picked = df.sample(n=min(N_PICK, len(df)), random_state=SEED)

# 5. 复制文件
os.makedirs(TARGET_DIR, exist_ok=True)
for fname in picked["filename"]:
    src = os.path.join(SOURCE_DIR, fname.strip())
    dst = os.path.join(TARGET_DIR, fname.strip())
    if os.path.isfile(src):
        shutil.copy2(src, dst)
    else:
        print(f"⚠ 找不到文件：{src}")

# 6. 保存新 csv
picked.to_csv(os.path.join(TARGET_DIR, "id_prop.csv"),
              sep="\t", index=False, float_format="%.6f")

print(f"✅ 已抽取 {len(picked)} 条数据到 {TARGET_DIR}")