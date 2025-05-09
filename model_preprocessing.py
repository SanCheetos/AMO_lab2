import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

train_dir = "train_processed"
train_files = [f for f in os.listdir(train_dir) if f.endswith(".csv")]

dataframes = []
for f in train_files:
    df = pd.read_csv(os.path.join(train_dir, f))
    if 'date' in df.columns and 'sales' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['day_num'] = df['date'].dt.day
        dataframes.append(df[['day_num', 'sales']])
    else:
        print(f"Пропущен файл без нужных столбцов: {f}")

all_data = pd.concat(dataframes)

X = all_data[['day_num']]
y = all_data['sales']

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "sales_model.pkl")
