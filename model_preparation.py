import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

os.makedirs("train_processed", exist_ok=True)
os.makedirs("test_processed", exist_ok=True)

def process_and_save(file_path, output_dir, scaler):
    df = pd.read_csv(file_path)
    
    if 'sales' in df.columns:
        df['sales'] = scaler.transform(df[['sales']])
    
    output_path = os.path.join(output_dir, os.path.basename(file_path))
    df.to_csv(output_path, index=False)

train_files = [f for f in os.listdir("train") if f.endswith(".csv")]
train_dfs = [pd.read_csv(os.path.join("train", f)) for f in train_files]
all_sales = pd.concat([df[['sales']] for df in train_dfs])

scaler = StandardScaler()
scaler.fit(all_sales)

for f in train_files:
    path = os.path.join("train", f)
    process_and_save(path, "train_processed", scaler)

test_files = [f for f in os.listdir("test") if f.endswith(".csv")]
for f in test_files:
    path = os.path.join("test", f)
    process_and_save(path, "test_processed", scaler)
