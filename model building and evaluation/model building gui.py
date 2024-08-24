import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import tkinter as tk
from tkinter import ttk

X_train = pd.read_csv('X_train.csv')
X_val = pd.read_csv('X_val.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').squeeze()
y_val = pd.read_csv('y_val.csv').squeeze()
y_test = pd.read_csv('y_test.csv').squeeze()

dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr_model.fit(X_train, y_train)

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    report = classification_report(y, y_pred, target_names=['Class 0', 'Class 1'])
    matrix = confusion_matrix(y, y_pred)
    return report, matrix

def create_gui():
    root = tk.Tk()
    root.title("Model Evaluation Results")
    root.geometry("1200x700")  

   
    main_frame = ttk.Frame(root, padding="10")
    main_frame.pack(fill=tk.BOTH, expand=True)

    
    title = ttk.Label(main_frame, text="Model Evaluation Results", font=("Helvetica", 18, "bold"))
    title.pack(pady=10)

   
    dt_frame = ttk.Frame(main_frame, padding="10", relief=tk.RAISED, borderwidth=2)
    dt_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)

    dt_title = ttk.Label(dt_frame, text="Decision Tree Evaluation", font=("Helvetica", 16, "bold"))
    dt_title.pack(pady=5)

    dt_text = tk.Text(dt_frame, wrap=tk.WORD, font=("Helvetica", 12), height=25, width=50)
    dt_text.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

    
    lr_frame = ttk.Frame(main_frame, padding="10", relief=tk.RAISED, borderwidth=2)
    lr_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=5)

    lr_title = ttk.Label(lr_frame, text="Logistic Regression Evaluation", font=("Helvetica", 16, "bold"))
    lr_title.pack(pady=5)

    lr_text = tk.Text(lr_frame, wrap=tk.WORD, font=("Helvetica", 12), height=25, width=50)
    lr_text.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

    def show_results():
    
        dt_val_report, dt_val_matrix = evaluate_model(dt_model, X_val, y_val)
        dt_test_report, dt_test_matrix = evaluate_model(dt_model, X_test, y_test)
        dt_results = (
            "=== Decision Tree Evaluation ===\n\n"
            "Validation Set:\n"
            f"{dt_val_report}\n"
            "Confusion Matrix:\n"
            f"{dt_val_matrix}\n\n"
            "Test Set:\n"
            f"{dt_test_report}\n"
            "Confusion Matrix:\n"
            f"{dt_test_matrix}\n"
        )
        dt_text.delete('1.0', tk.END)
        dt_text.insert(tk.END, dt_results)

    
        lr_val_report, lr_val_matrix = evaluate_model(lr_model, X_val, y_val)
        lr_test_report, lr_test_matrix = evaluate_model(lr_model, X_test, y_test)
        lr_results = (
            "=== Logistic Regression Evaluation ===\n\n"
            "Validation Set:\n"
            f"{lr_val_report}\n"
            "Confusion Matrix:\n"
            f"{lr_val_matrix}\n\n"
            "Test Set:\n"
            f"{lr_test_report}\n"
            "Confusion Matrix:\n"
            f"{lr_test_matrix}\n"
        )
        lr_text.delete('1.0', tk.END)
        lr_text.insert(tk.END, lr_results)


    eval_button = ttk.Button(main_frame, text="Evaluate Models", command=show_results)
    eval_button.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    create_gui()
