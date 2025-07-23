import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load('final_gradient_boosting_model.pkl')
scaler = joblib.load('scaler.pkl') 
# Load any encoders if used (e.g., OneHotEncoder, LabelEncoder)
# encoder = joblib.load('encoder.pkl')  # optional, if you saved one

# Streamlit App Title
st.title("Employee Salary Prediction App")
st.subheader("Will the employee earn >50K or <=50K?")

# Input form
def user_input_features():
    age = st.slider("Age", 17, 90, 30)
    workclass = st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
                                           'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])
    education = st.selectbox("Education", ['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
                                           'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school'])
    marital_status = st.selectbox("Marital Status", ['Never-married', 'Married-civ-spouse', 'Divorced',
                                                     'Married-spouse-absent', 'Separated', 'Widowed'])
    occupation = st.selectbox("Occupation", ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial',
                                             'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical',
                                             'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv',
                                             'Armed-Forces'])
    relationship = st.selectbox("Relationship", ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])
    race = st.selectbox("Race", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
    sex = st.selectbox("Sex", ['Male', 'Female'])
    capital_gain = st.number_input("Capital Gain", min_value=0)
    capital_loss = st.number_input("Capital Loss", min_value=0)
    hours_per_week = st.slider("Hours per Week", 1, 99, 40)
    native_country = st.selectbox("Native Country", ['United-States', 'Mexico', 'Philippines', 'Germany', 'Canada',
                                                     'India', 'England', 'Cuba', 'Jamaica', 'China', 'Others'])

    data = {
        'age': age,
        'workclass': workclass,
        'education': education,
        'marital-status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'sex': sex,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week,
        'native-country': native_country
    }

    return pd.DataFrame([data])

# Get user input
input_df = user_input_features()
# Log-transform capital gain and loss to match training preprocessing
input_df['capital-gain'] = np.log1p(input_df['capital-gain'])
input_df['capital-loss'] = np.log1p(input_df['capital-loss'])
# Apply encoding to match the training data
# If you used OneHotEncoder and saved it:
# input_df_encoded = encoder.transform(input_df)
# If you manually used pd.get_dummies, recreate it here:
input_encoded = pd.get_dummies(input_df)

# Align columns with training set (ensure same order & shape)
model_columns = joblib.load("model_columns.pkl")  # Save this during training
input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
input_scaled=scaler.transform(input_encoded)

# Prediction
prediction = model.predict(input_encoded)
pred_proba = model.predict_proba(input_encoded)

# Output''

if prediction[0] == 1:
    st.success(f"Prediction: Income >50K  (Confidence: {round(pred_proba[0][1]*100, 2)}%)")
else:
    st.warning(f"Prediction: Income <=50K (Confidence: {round(pred_proba[0][0]*100, 2)}%)")