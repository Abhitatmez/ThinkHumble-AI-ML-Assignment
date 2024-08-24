import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump
import tkinter as tk
from tkinter import scrolledtext


X_train = pd.read_csv('X_train.csv')
X_val = pd.read_csv('X_val.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').squeeze()  # Convert DataFrame to Series
y_val = pd.read_csv('y_val.csv').squeeze()
y_test = pd.read_csv('y_test.csv').squeeze()


dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr_model.fit(X_train, y_train)


dump(dt_model, 'decision_tree_model.joblib')
dump(lr_model, 'logistic_regression_model.joblib')


def evaluate_model(model, X, y, model_name):
    y_pred = model.predict(X)
    report = classification_report(y, y_pred)
    matrix = confusion_matrix(y, y_pred)
    return f"{model_name} Performance:\n{report}\nConfusion Matrix:\n{matrix}"

root = tk.Tk()
root.title("Model Evaluation Results")

text_area_dt = scrolledtext.ScrolledText(root, width=60, height=20)
text_area_dt.grid(row=0, column=0, padx=10, pady=10)

text_area_lr = scrolledtext.ScrolledText(root, width=60, height=20)
text_area_lr.grid(row=0, column=1, padx=10, pady=10)

dt_eval_val = evaluate_model(dt_model, X_val, y_val, "Decision Tree (Validation)")
lr_eval_val = evaluate_model(lr_model, X_val, y_val, "Logistic Regression (Validation)")

dt_eval_test = evaluate_model(dt_model, X_test, y_test, "Decision Tree (Test)")
lr_eval_test = evaluate_model(lr_model, X_test, y_test, "Logistic Regression (Test)")

text_area_dt.insert(tk.END, f"Validation Results:\n{dt_eval_val}\n\nTest Results:\n{dt_eval_test}")
text_area_lr.insert(tk.END, f"Validation Results:\n{lr_eval_val}\n\nTest Results:\n{lr_eval_test}")


with open('evaluation_results.txt', 'w') as file:
    file.write(f"Decision Tree Evaluation:\n{dt_eval_val}\n\nLogistic Regression Evaluation:\n{lr_eval_val}\n\n")
    file.write(f"Decision Tree Final Testing:\n{dt_eval_test}\n\nLogistic Regression Final Testing:\n{lr_eval_test}")

root.mainloop()
