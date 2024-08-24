
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


df = pd.read_csv("synthetic_telecom_data.csv")

window = tk.Tk()
window.title("EDA and Data Preprocessing")

frame_text = tk.Frame(window)
frame_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

frame_plots = tk.Frame(window)
frame_plots.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

text_area = scrolledtext.ScrolledText(frame_text, wrap=tk.WORD, width=60, height=20)
text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)


class RedirectText:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, text):
        self.text_widget.insert(tk.END, text)
        self.text_widget.see(tk.END)

    def flush(self):
        pass

import sys
sys.stdout = RedirectText(text_area)
sys.stderr = RedirectText(text_area)

print("Data Overview")
print(df.head())
print("\nData Info")
print(df.info())
print("\nData Description")
print(df.describe())

fig_age = plt.Figure(figsize=(6, 4), dpi=100)
ax_age = fig_age.add_subplot(111)
sns.histplot(df['Age'], kde=True, ax=ax_age)
ax_age.set_title('Age Distribution')
canvas_age = FigureCanvasTkAgg(fig_age, master=frame_plots)
canvas_age.draw()
canvas_age.get_tk_widget().pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

fig_churn = plt.Figure(figsize=(6, 4), dpi=100)
ax_churn = fig_churn.add_subplot(111)
sns.countplot(x='Churn', data=df, ax=ax_churn)
ax_churn.set_title('Churn Distribution')
canvas_churn = FigureCanvasTkAgg(fig_churn, master=frame_plots)
canvas_churn.draw()
canvas_churn.get_tk_widget().pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

df.fillna(df.mean(numeric_only=True), inplace=True)  # Impute missing values
df.loc[df['MonthlyCharges'] < 0, 'MonthlyCharges'] = df['MonthlyCharges'].median()  # Correct invalid values

df = pd.get_dummies(df, columns=['Gender', 'ContractType', 'TechSupport', 'InternetService', 'PaperlessBilling', 'PaymentMethod'])

X = df.drop(['CustomerID', 'Churn'], axis=1)
y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)  # Convert 'Yes'/'No' to 1/0

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print("\nData preprocessing complete.")

window.mainloop()
