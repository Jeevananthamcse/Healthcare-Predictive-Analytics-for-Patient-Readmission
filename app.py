from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the model and label encoders
with open('model.pkl', 'rb') as file:
    model, label_encoders, columns = pickle.load(file)

# Function to handle unseen labels
def transform_with_unseen(le, value):
    if value in le.classes_:
        return le.transform([value])[0]
    else:
        new_classes = np.append(le.classes_, value)
        le.classes_ = new_classes
        return le.transform([value])[0]

@app.route('/')
def home():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_features = {
        'age': request.form['age'],
        'time_in_hospital': int(request.form['time_in_hospital']),
        'n_lab_procedures': int(request.form['n_lab_procedures']),
        'n_procedures': int(request.form['n_procedures']),
        'n_medications': int(request.form['n_medications']),
        'n_outpatient': int(request.form['n_outpatient']),
        'n_inpatient': int(request.form['n_inpatient']),
        'n_emergency': int(request.form['n_emergency']),
        'medical_specialty': request.form['medical_specialty'],
        'diag_1': request.form['diag_1'],
        'diag_2': request.form['diag_2'],
        'diag_3': request.form['diag_3'],
        'glucose_test': request.form['glucose_test'],
        'A1Ctest': request.form['A1Ctest'],
        'change': request.form['change'],
        'diabetes_med': request.form['diabetes_med']
    }

    input_df = pd.DataFrame([input_features], columns=columns)
    for column, le in label_encoders.items():
        input_df[column] = input_df[column].apply(lambda x: transform_with_unseen(le, x))

    prediction = model.predict(input_df)
    result = 'Yes' if prediction[0] == 1 else 'No'

    return render_template('output.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
