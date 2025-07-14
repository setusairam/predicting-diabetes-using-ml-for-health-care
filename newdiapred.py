import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC

# Load the data
data = pd.read_csv('E:\CSP\project\diabetes.csv')

# Split the data into features and target
x = data.drop(['Outcome'], axis=1)
y = data['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Standardize the data (you can also try MinMaxScaler)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Try different logistic regression parameters with GridSearchCV
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

# Using GridSearchCV to find the best logistic regression parameters
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print(f"Best Logistic Regression parameters: {grid.best_params_}")

# Evaluate the best logistic regression model
best_logreg_model = grid.best_estimator_
y_pred_logreg = best_logreg_model.predict(X_test)

# Calculate accuracy and R2 score for Logistic Regression
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
r2_logreg = r2_score(y_test, y_pred_logreg)
print(f"Logistic Regression Accuracy: {accuracy_logreg * 100:.2f}%")
print(f"Logistic Regression R2 Score: {r2_logreg}")

# Cross-validation score for logistic regression
cross_val_logreg = cross_val_score(best_logreg_model, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-validated Logistic Regression accuracy: {np.mean(cross_val_logreg) * 100:.2f}%")

# Try RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Calculate accuracy for Random Forest
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf * 100:.2f}%")

# Try creating Polynomial Features
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Train logistic regression on polynomial features
poly_model = LogisticRegression(C=1.0, solver='liblinear')
poly_model.fit(X_train_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly)

# Calculate accuracy for polynomial logistic regression
accuracy_poly = accuracy_score(y_test, y_pred_poly)
print(f"Polynomial Logistic Regression Accuracy: {accuracy_poly * 100:.2f}%")

# Ensemble: Logistic Regression, Random Forest, and SVM
clf1 = LogisticRegression(C=1.0, solver='liblinear')
clf2 = RandomForestClassifier(n_estimators=100)
clf3 = SVC(probability=True)

ensemble_model = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('svc', clf3)], voting='soft')
ensemble_model.fit(X_train, y_train)
y_pred_ensemble = ensemble_model.predict(X_test)

# Calculate accuracy for the ensemble model
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
print(f"Ensemble Model Accuracy: {accuracy_ensemble * 100:.2f}%")

# Test with a new data point
new_data_point = [[6, 148, 72, 35, 94, 33.6, 0.627, 50]]
new_data_df = pd.DataFrame(new_data_point, columns=x.columns)

# Make a prediction for the new data point using the ensemble model
prediction = ensemble_model.predict(sc.transform(new_data_df))

if prediction[0] == 0:
    print("The person is not diabetic")
else:
    print("The person is diabetic")
