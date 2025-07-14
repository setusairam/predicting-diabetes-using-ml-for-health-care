from flask import Flask, request, render_template, redirect, url_for, session
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Load the dataset
data = pd.read_csv('E:/CSP/project/diabetes.csv')
x = data.drop(['Outcome'], axis=1)
y = data['Outcome']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Standardize the dataset
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train the model
model = RandomForestClassifier()
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
    prediction = model.predict(sc.transform(new_data_df))

    # Prepare prediction result
    if prediction[0] == 0:
        result = "The person is not diabetic"
    else:
        result = "The person is diabetic"

    return render_template('predictor.html', prediction_text=result)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

if __name__ == "__main__":
    app.run(debug=True)
