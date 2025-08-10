import pandas as pd
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__, template_folder='Flask/templates', static_folder='Flask/static')

import os

pkl_folder = 'Training/'

num_imputer = joblib.load(os.path.join(pkl_folder, 'num_imputer.pkl'))
scaler = joblib.load(os.path.join(pkl_folder, 'scaler.pkl'))
cat_imputer = joblib.load(os.path.join(pkl_folder, 'cat_imputer.pkl'))
onehot_encoder = joblib.load(os.path.join(pkl_folder, 'onehot_encoder.pkl'))
model = joblib.load(os.path.join(pkl_folder, 'model.pkl'))
cols_info = joblib.load(os.path.join(pkl_folder, 'cols_info.pkl'))


# cols_info = joblib.load('cols_info.pkl')
num_cols = cols_info['num_cols']
cat_cols = cols_info['cat_cols']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/form1')
def form1():
    return render_template('index1.html')

@app.route('/form2')
def form2():
    return render_template('indexnew.html')

# Mapping user input names to dataset columns
user_to_dataset_map = {
    'blood_urea': 'bu',
    'blood_glucose_random': 'bgr',
    'anemia': 'ane',
    'coronary_artery_disease': 'cad',
    'pus_cell': 'pc',
    'red_blood_cell': 'rbc',
    'diabetes_mellitus': 'dm',
    'pedal_edema': 'pe'
}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_dict = {}
        default_values = {
    'id': '0',
    'age': '50',  # median age example
    'bp': '80',
    'sg': '1.020',
    'al': '0',
    'su': '0',
    'rbc': 'normal',
    'pc': 'normal',
    'pcc': 'notpresent',
    'ba': 'notpresent',
    'bgr': '100',  # blood glucose random
    'bu': '30',    # blood urea
    'sc': '1.0',
    'sod': '140',
    'pot': '4.5',
    'hemo': '15',
    'pcv': '44',
    'wc': '8000',
    'rc': '5',
    'htn': 'no',
    'dm': 'no',     # diabetes mellitus
    'cad': 'no',    # coronary artery disease
    'appet': 'good',
    'pe': 'no',     # pedal edema
    'ane': 'no'     # anemia
}


        # Start by filling all features with default values
        for col in num_cols + cat_cols:
            input_dict[col] = [default_values.get(col, '')]  # fallback to empty string if no default

        # Now override those with user inputs for the 8 features
        for user_key, dataset_col in user_to_dataset_map.items():
            val = request.form.get(user_key)
            if val is not None and val.strip() != '':
                input_dict[dataset_col] = [val]
            else:
                # if no input, keep default (already set)
                pass

        X_input = pd.DataFrame(input_dict)

        # Continue preprocessing as you have
        X_num = X_input[num_cols].apply(pd.to_numeric, errors='coerce')
        X_num_imputed = num_imputer.transform(X_num)
        X_num_scaled = scaler.transform(X_num_imputed)

        X_cat = X_input[cat_cols].fillna('')
        X_cat_imputed = cat_imputer.transform(X_cat)
        X_cat_encoded = onehot_encoder.transform(X_cat_imputed)

        X_preprocessed = np.hstack([X_num_scaled, X_cat_encoded])

        pred = model.predict(X_preprocessed)[0]
        proba = model.predict_proba(X_preprocessed)[0][1] if hasattr(model, 'predict_proba') else None

        label = 'CKD Detected' if float(pred) == 1.0 else 'No CKD Detected'
        probability = round(float(proba), 3) if proba is not None else 'N/A'

        # return jsonify({'prediction_text': label, 'probability': probability})
        return render_template('result.html', prediction_text=label, probability=probability)

    except Exception as e:
        # return jsonify({'prediction_text': f"Error: {str(e)}", 'probability': 'N/A'})
        return render_template('result.html', prediction_text=f"Error: {str(e)}", probability='N/A')


if __name__ == "__main__":
    app.run(debug=True)
