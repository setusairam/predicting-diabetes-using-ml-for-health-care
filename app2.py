from flask import Flask, request, render_template, redirect, url_for, session
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Load the dataset
try:
    data = pd.read_csv('E:/CSP/project/diabetes.csv')
except FileNotFoundError:
    raise FileNotFoundError("Dataset 'diabetes.csv' not found at the specified path.")

# Data cleaning: replace empty strings or spaces with NaN
data = data.replace(r'^\s*$', np.nan, regex=True)

# Separate features and target
x = data.drop(['Outcome'], axis=1)
y = data['Outcome']

# Check for NaN values
if x.isnull().values.any():
    print("Data contains NaN values. Imputing missing values...")

# Handle missing values using SimpleImputer (fill with mean)
imputer = SimpleImputer(strategy='mean')
x = pd.DataFrame(imputer.fit_transform(x), columns=x.columns)

# Ensure all data is numeric
x = x.apply(pd.to_numeric, errors='coerce')

# Split the dataset
try:
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
except ValueError as e:
    raise ValueError(f"Error splitting data: {str(e)}")

# Standardize the dataset
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train the model
model = RandomForestClassifier(random_state=42)  # Set random state for reproducibility
model.fit(X_train, y_train)

# Routes
@app.route('/')
def login():
    return render_template('test.html')

@app.route('/login', methods=['POST'])
def do_login():
    username = request.form['username']
    password = request.form['password']
    
    if username == 'admin' and password == 'password':
        session['logged_in'] = True
        return redirect(url_for('home'))
    else:
        return render_template('test.html', error="Invalid credentials. Please try again.")

@app.route('/home')
def home():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/predictor')
def predictor():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('predictor.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    try:
        # Collect data from the form and map it to the corresponding columns
        input_data = [float(request.form['pregnancies']),
                      float(request.form['glucose']),
                      float(request.form['bloodpressure']),
                      float(request.form['skinthickness']),
                      float(request.form['insulin']),
                      float(request.form['bmi']),
                      float(request.form['dpf']),
                      float(request.form['age'])]

        # Prepare input data for prediction
        new_data_df = pd.DataFrame([input_data], columns=x.columns)
        # Impute missing values in input data if any
        new_data_df = pd.DataFrame(imputer.transform(new_data_df), columns=x.columns)
        prediction = model.predict(sc.transform(new_data_df))

        # Prepare prediction result
        if prediction[0] == 0:
            result = "The person is not diabetic"
        else:
            result = "The person is diabetic"
    except ValueError as e:
        result = f"Error in input data: {str(e)}"

    return render_template('predictor.html', prediction_text=result)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

if __name__ == "__main__":
    app.run(debug=True)
