import pandas as pd
import numpy as np
from faker import Faker
import random

fake = Faker()

random.seed(42)
np.random.seed(42)
Faker.seed(42)

num_records = 5000

data = {
    "CustomerID": [fake.uuid4() for _ in range(num_records)],
    "Age": np.random.randint(18, 80, size=num_records),
    "Gender": np.random.choice(["Male", "Female"], size=num_records),
    "ContractType": np.random.choice(["Month-to-month", "One year", "Two year"], size=num_records),
    "MonthlyCharges": np.round(np.random.uniform(20, 120, size=num_records), 2),
    "TotalCharges": np.round(np.random.uniform(500, 5000, size=num_records), 2),
    "TechSupport": np.random.choice(["Yes", "No"], size=num_records),
    "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], size=num_records),
    "Tenure": np.random.randint(1, 72, size=num_records),  # Tenure in months
    "PaperlessBilling": np.random.choice(["Yes", "No"], size=num_records),
    "PaymentMethod": np.random.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], size=num_records),
    "Churn": np.random.choice(["Yes", "No"], p=[0.2, 0.8], size=num_records)
}


df = pd.DataFrame(data)

df['TotalCharges'] = df['MonthlyCharges'] * df['Tenure']

df['Average_Monthly_Charges'] = df['TotalCharges'] / np.where(df['Tenure'] == 0, 1, df['Tenure'])

df['Customer_Lifetime_Value'] = df['Average_Monthly_Charges'] * df['Tenure'] * 10

for col in ['Age', 'MonthlyCharges', 'TotalCharges']:
    df.loc[df.sample(frac=0.05).index, col] = np.nan  # 5% missing data

df.loc[df.sample(frac=0.01).index, 'MonthlyCharges'] = -10  # 1% wrong values

df.to_csv("synthetic_telecom_data.csv", index=False)

print("Synthetic data generated successfully!")
