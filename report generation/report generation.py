
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from joblib import load
from fpdf import FPDF

X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv').squeeze()

if 'CustomerID' in X_test.columns:
    customer_ids = X_test[['CustomerID']]
    X_test = X_test.drop(columns=['CustomerID'])
else:
    customer_ids = pd.DataFrame({'CustomerID': range(len(X_test))})

dt_model = load('decision_tree_model.joblib')
lr_model = load('logistic_regression_model.joblib')

dt_predictions = dt_model.predict(X_test)
lr_predictions = lr_model.predict(X_test)

predictions_df = pd.DataFrame({
    'CustomerID': customer_ids['CustomerID'],
    'Predicted_Churn_DT': dt_predictions,
    'Predicted_Churn_LR': lr_predictions
})

predictions_df.to_csv('predicted_churn.csv', index=False)

def evaluate_model(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    confusion = confusion_matrix(y_true, y_pred)
    return report, confusion

dt_report, dt_confusion = evaluate_model(y_test, dt_predictions)
lr_report, lr_confusion = evaluate_model(y_test, lr_predictions)

def forecast_churn(model, X):
    predictions = model.predict(X)
    churn_rate = predictions.mean() * 100  
    return churn_rate

dt_churn_rate = forecast_churn(dt_model, X_test)
lr_churn_rate = forecast_churn(lr_model, X_test)


class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Customer Churn Prediction Report', 0, 1, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

    def add_chart(self, chart_path, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)
        self.image(chart_path, x=10, w=180)
        self.ln()

pdf = PDF()
pdf.add_page()


pdf.chapter_title('1. Exploratory Data Analysis Insights')
pdf.chapter_body("The dataset provides information about customer demographics, service usage, and contract details. Key insights include:"
                 "\n- The average age of customers is around 50 years."
                 "\n- Monthly charges and total charges vary significantly among customers."
                 "\n- There are distinct patterns in churn rates based on contract types and tech support usage.")


pdf.chapter_title('2. Model Performance')
pdf.chapter_body("We evaluated two models: Decision Tree and Logistic Regression. Here are the key findings for each model:"
                 "\n\nDecision Tree:"
                 f"\n\n{classification_report(y_test, dt_predictions)}"
                 "\nConfusion Matrix:"
                 f"\n{dt_confusion}"
                 "\n\nLogistic Regression:"
                 f"\n\n{classification_report(y_test, lr_predictions)}"
                 "\nConfusion Matrix:"
                 f"\n{lr_confusion}")


pdf.chapter_title('3. Future Churn Prediction')
pdf.chapter_body(f"The Decision Tree model predicts a churn rate of approximately {dt_churn_rate:.2f}%."
                 f"\nThe Logistic Regression model predicts a churn rate of approximately {lr_churn_rate:.2f}%.")

def plot_roc_curve(model, X, y, model_name):
    from sklearn.metrics import roc_curve, auc

    y_probs = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve ({model_name})')
    plt.legend(loc='lower right')
    plt.savefig(f'roc_curve_{model_name.lower().replace(" ", "_")}.png')
    plt.close()

plot_roc_curve(dt_model, X_test, y_test, "Decision Tree")
plot_roc_curve(lr_model, X_test, y_test, "Logistic Regression")


pdf.chapter_title('4. ROC Curves')
pdf.add_chart('roc_curve_decision_tree.png', 'ROC Curve - Decision Tree')
pdf.add_chart('roc_curve_logistic_regression.png', 'ROC Curve - Logistic Regression')

pdf.output("Customer_Churn_Prediction_Report.pdf")

print("Report generation complete.")
