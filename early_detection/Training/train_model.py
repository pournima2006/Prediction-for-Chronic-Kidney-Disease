import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np

import pandas as pd

# Load your dataset CSV file (adjust path and filename accordingly)
df = pd.read_csv('../dataset/chronickidneydisease.csv')


# Define your feature columns and target column
num_cols = ['age', 'bp', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
cat_cols = ['sg', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']

feature_cols = num_cols + cat_cols

# Replace known invalid entries with NaN
df.replace(['?', '\t?', '\t'], np.nan, inplace=True)

# Convert numeric columns explicitly to numeric (coerce errors to NaN)
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Now extract features again (after cleaning)
X = df[feature_cols]
y = df['classification'].apply(lambda x: 1 if x == 'ckd' else 0)

X_num = X[num_cols]
X_cat = X[cat_cols]

# Proceed with imputing and scaling


# Numeric preprocessing
num_imputer = SimpleImputer(strategy='median')
X_num_imputed = num_imputer.fit_transform(X_num)

scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num_imputed)

# Categorical preprocessing
cat_imputer = SimpleImputer(strategy='most_frequent')
X_cat_imputed = cat_imputer.fit_transform(X_cat)

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_cat_encoded = ohe.fit_transform(X_cat_imputed)

# Combine numeric and categorical
X_preprocessed = np.hstack([X_num_scaled, X_cat_encoded])

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_preprocessed, y)

# Save preprocessing objects and model separately
joblib.dump(num_imputer, 'num_imputer.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(cat_imputer, 'cat_imputer.pkl')
joblib.dump(ohe, 'onehot_encoder.pkl')
joblib.dump(model, 'model.pkl')

# Save column info (optional)
joblib.dump({'num_cols': num_cols, 'cat_cols': cat_cols}, 'cols_info.pkl')

print("Training complete and all artifacts saved.")
