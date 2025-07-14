import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# Load the data
data = pd.read_csv('E:\CSP\project\diabetes.csv')

# Split the data into features and target
x = data.drop(['Outcome'], axis=1)
y = data['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Standardize the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Calculate the R2 score (not usually used for classification)
r2 = r2_score(y_test, y_pred)
print(f"R2 score: {r2}")

# Test with a new data point
new_data_point = [[6, 148, 72, 35, 94, 33.6, 0.627, 50]]
new_data_df = pd.DataFrame(new_data_point, columns=x.columns)

# Make a prediction for the new data point
prediction = model.predict(sc.transform(new_data_df))

if prediction[0] == 0:
    print("The person is not diabetic")
else:
    print("The person is diabetic")
