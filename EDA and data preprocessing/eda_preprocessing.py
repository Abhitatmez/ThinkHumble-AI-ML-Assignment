
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("synthetic_telecom_data.csv")

print(df.head())
print(df.info())
print(df.describe())

sns.histplot(df['Age'], kde=True)
plt.title('Age Distribution')
plt.show()

sns.countplot(x='Churn', data=df)
plt.title('Churn Distribution')
plt.show()

df.fillna(df.mean(numeric_only=True), inplace=True)  # Impute missing values
df.loc[df['MonthlyCharges'] < 0, 'MonthlyCharges'] = df['MonthlyCharges'].median()  # Correct invalid values

df = pd.get_dummies(df, columns=['Gender', 'ContractType', 'TechSupport', 'InternetService', 'PaperlessBilling', 'PaymentMethod'])

X = df.drop(['CustomerID', 'Churn'], axis=1)
y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)  

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

X_train.to_csv('X_train.csv', index=False)
X_val.to_csv('X_val.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_val.to_csv('y_val.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print("Data preprocessing complete and saved.")
