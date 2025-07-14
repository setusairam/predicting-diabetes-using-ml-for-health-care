from sklearn.metrics import confusion_matrix

# Confusion Matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# Create a detailed performance table
performance_metrics = pd.DataFrame({
    "Metric": ["True Positives", "True Negatives", "False Positives", "False Negatives",
               "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
    "Value": [tp, tn, fp, fn,
              accuracy, precision, recall, f1, roc_auc]
})
print(performance_metrics)
