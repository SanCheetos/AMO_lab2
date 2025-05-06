import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)
os.makedirs("train", exist_ok=True)
os.makedirs("test", exist_ok=True)

def generate_sales_data(start_date, days=30, base_sales=100, noise_level=5.0, anomaly=False):
    dates = [start_date + timedelta(days=i) for i in range(days)]
    base_curve = base_sales + 20 * np.sin(np.linspace(0, 2 * np.pi, days))  # сезонность (неделя/месяц)
    noise = np.random.normal(0, noise_level, size=days)
    sales = base_curve + noise
    sales = np.clip(sales, 0, None)  # продажи не могут быть отрицательными

    if anomaly:
        num_anomalies = np.random.randint(1, 4)
        for _ in range(num_anomalies):
            idx = np.random.randint(0, days)
            sales[idx] += np.random.choice([-40, 50])

    df = pd.DataFrame({
        'date': dates,
        'sales': sales.round(2)
    })

    return df

#Generating
for i in range(5):
    df = generate_sales_data(datetime(2025, 1, 1), anomaly=(i % 2 == 0), noise_level=5 + i*1.5)
    df.to_csv(f"train/sales_train_{i}.csv", index=False)

for i in range(3):
    df = generate_sales_data(datetime(2025, 2, 1), anomaly=(i % 2 == 1), noise_level=6 + i*1.5)
    df.to_csv(f"test/sales_test_{i}.csv", index=False)