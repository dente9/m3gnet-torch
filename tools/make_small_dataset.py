# make_small_dataset.py  —— 只改这里，训练脚本零改动
import os
import shutil
import random
import pandas as pd

SOURCE_DIR = 'data/aaa'
TARGET_DIR = 'data/bbb'
N_PICK   = 2000
SEED     = 42

random.seed(SEED)

csv_path = os.path.join(SOURCE_DIR, 'id_prop.csv')

# 1. 自动探测并拆成两列
df = pd.read_csv(csv_path, sep=None, engine='python')   # 自动识别逗号/制表符
if df.shape[1] == 1:                                    # 只有一列 → 手动拆分
    df = df.iloc[:, 0].str.strip().str.split(r'[,\t]', expand=True)
df.columns = ['filename', 'property']
df['property'] = pd.to_numeric(df['property'], errors='coerce')

# 2. 随机抽取
picked = df.sample(n=min(N_PICK, len(df)), random_state=SEED)

# 3. 复制 cif
os.makedirs(TARGET_DIR, exist_ok=True)
for fname in picked['filename']:
    src = os.path.join(SOURCE_DIR, fname.strip())
    dst = os.path.join(TARGET_DIR, fname.strip())
    if os.path.isfile(src):
        shutil.copy2(src, dst)
    else:
        print(f'⚠ 缺失文件：{src}')

# 4. 保存为逗号分隔，训练脚本无需改
picked.to_csv(os.path.join(TARGET_DIR, 'id_prop.csv'),
              sep=',', index=False, float_format='%.6f')

print(f'✅ 已抽取 {len(picked)} 条数据到 {TARGET_DIR}')