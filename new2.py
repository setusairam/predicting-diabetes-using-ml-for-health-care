import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
data = pd.read_csv('E:/CSP/project/diabetes.csv')

# Data Cleaning
print(data.isnull().sum())

# 2. Remove duplicates (if any)
data.drop_duplicates(inplace=True)

# 3. Remove outliers using Z-score (for demonstration, threshold set to 3)
z_scores = np.abs((data - data.mean()) / data.std())
data = data[(z_scores < 3).all(axis=1)]

# 4. Check for highly correlated features
corr_matrix = data.corr().abs()

# Remove highly correlated features (threshold > 0.9)
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
data.drop(to_drop, axis=1, inplace=True)

# Split the data into features and target
X = data.drop(['Outcome'], axis=1)
y = data['Outcome']

# Handle class imbalance using SMOTE
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Standardize the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Set up a hyperparameter grid for Logistic Regression and RandomForest
param_grid_lr = [
    {
        'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
        'solver': ['lbfgs'],
        'penalty': ['l2'],             # lbfgs only supports 'l2' penalty
    },
    {
        'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
        'solver': ['liblinear'],
        'penalty': ['l1', 'l2'],       # liblinear supports both 'l1' and 'l2' penalties
    }
]

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Initialize models
log_reg = LogisticRegression(max_iter=1000)
rf_model = RandomForestClassifier(random_state=42)

# Perform grid search for Logistic Regression
grid_search_lr = GridSearchCV(estimator=log_reg, param_grid=param_grid_lr, cv=5, scoring='accuracy')
grid_search_lr.fit(X_train, y_train)

# Perform grid search for Random Forest
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=5, scoring='accuracy')
grid_search_rf.fit(X_train, y_train)

# Retrieve the best models
best_log_reg = grid_search_lr.best_estimator_
best_rf_model = grid_search_rf.best_estimator_

# Evaluate both models
y_pred_lr = best_log_reg.predict(X_test)
y_pred_rf = best_rf_model.predict(X_test)

accuracy_lr = accuracy_score(y_test, y_pred_lr)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

print(f"Logistic Regression Accuracy: {accuracy_lr * 100:.2f}%")
print(f"Random Forest Accuracy: {accuracy_rf * 100:.2f}%")

# Print classification reports
print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_lr))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

# Test with a new data point
new_data_point = [[6, 148, 72, 35, 94, 33.6, 0.627, 50]]
new_data_df = pd.DataFrame(new_data_point, columns=X.columns)

# Make a prediction for the new data point
new_data_scaled = sc.transform(new_data_df)
prediction = best_log_reg.predict(new_data_scaled) if accuracy_lr > accuracy_rf else best_rf_model.predict(new_data_scaled)

if prediction[0] == 0:
    print("The person is not diabetic")
else:
    print("The person is diabetic")
