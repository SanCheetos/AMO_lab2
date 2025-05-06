import os
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import joblib

model = joblib.load("sales_model.pkl")

test_dir = "test_processed"
test_files = [f for f in os.listdir(test_dir) if f.endswith(".csv")]

dataframes = []
for f in test_files:
    df = pd.read_csv(os.path.join(test_dir, f))
    if 'date' in df.columns and 'sales' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['day_num'] = df['date'].dt.day
        dataframes.append(df[['day_num', 'sales']])
    else:
        print(f"Пропущен файл без нужных столбцов: {f}")

all_data = pd.concat(dataframes)

X_test = all_data[['day_num']]
y_true = all_data['sales']
y_pred = model.predict(X_test)

mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f" - MSE: {mse:.3f}")
print(f" - R^2: {r2:.3f}")