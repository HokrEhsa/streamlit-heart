#!/usr/bin/env python
# coding: utf-8

# In[1]:

# Import modules
import pickle          # To load prediction model
import pandas as pd    # To load dataset
import streamlit as st # To create streamlit formatting
import numpy as np     # To round the percentage probability for better viewing

# Add title
st.write('''
# Heart Disease Prediction App
This application will predict the possibility of having indicators of **Coronary Heart Disease**.
''')

# Sidebar components
st.sidebar.image('heartbeat_transbg_logo.png', use_column_width=True)

st.sidebar.header("User Input Parameters")

# Import datasets for showcase
heart_bfmodel = pd.read_csv("heart.csv")
heart = pd.read_csv("heart_le.csv")

# Import model used for predicting heart disease
with open('RFC_model.pkl', 'rb') as file:
    RFC_model = pickle.load(file)

# Inputting parameters
def user_input_features():
    age = st.sidebar.slider("Age", 18, 100, 30)
    sex = st.sidebar.radio("Sex", ['Male', 'Female'])
    cptype = st.sidebar.radio("Chest Pain Type", ['Asymptomatic', 'Non-Anginal Pain', 'Atypical Angina', 'Typical Angina'])
    restbp = st.sidebar.slider("Resting Blood Pressure", 80, 200, 120)
    chol = st.sidebar.slider("Total Cholesterol Level", 80, 610, 250)
    fastbs = st.sidebar.radio("Fasting Blood Sugar", ['> 120 mg/dL', 'Otherwise'])
    restecg = st.sidebar.radio("Resting Electrocardiography", ['Normal', 'Left Ventricular Hypertrophy', 'ST-T Wave Abnormality'])
    maxhr = st.sidebar.slider("Maximum Heart Rate", 60, 205, 175)
    exang = st.sidebar.radio("Exercise-Induced Angina", ['No', 'Yes'])
    oldpk = st.sidebar.slider("Old Peak", -3.0, 6.5, 0.0)
    slope = st.sidebar.radio("ST Slope", ['Downsloping', 'Flat', 'Upsloping'])
    
    # Store the inputs in a Dataframe
    data = {'Age': age,
            'Sex': sex,
            'ChestPainType': cptype,
            'RestingBP': restbp,
            'Cholesterol': chol,
            'FastingBS': fastbs,
            'RestingECG': restecg,
            'MaxHR': maxhr,
            'ExerciseAngina': exang,
            'Oldpeak': oldpk,
            'ST_Slope': slope
           }
    
    features = pd.DataFrame(data, index=[0])
    
    return features

# Call function
df = user_input_features()

# Scaling and encoding dataframe of inputted parameters
df_scale = df.copy(deep=True)

def scale_dataframe():
    
    # Each of the features are dealt with differently
    
    # Numerical features except Oldpeak are standardised using the mean and standard deviation values from the prepared dataset
    df_scale['Age'] = (df_scale['Age'] - 53.509269) / 9.437636
    
    # Categorical features are encoded with labels for each value following the prepared dataset
    if df_scale.loc[0, 'Sex'] == 'Male':
            df_scale.loc[0, 'Sex'] = 1
    elif df_scale.loc[0, 'Sex'] == 'Female':
            df_scale.loc[0, 'Sex'] = 0

    if df_scale.loc[0, 'ChestPainType'] == 'Asymptomatic':
            df_scale.loc[0, 'ChestPainType'] = 0
    elif df_scale.loc[0, 'ChestPainType'] == 'Atypical Angina':
            df_scale.loc[0, 'ChestPainType'] = 1   
    elif df_scale.loc[0, 'ChestPainType'] == 'Non-Anginal Pain':
            df_scale.loc[0, 'ChestPainType'] = 2
    elif df_scale.loc[0, 'ChestPainType'] == 'Typical Angina':
            df_scale.loc[0, 'ChestPainType'] = 3

    df_scale['RestingBP'] = (df_scale['RestingBP'] - 132.540894) / 17.999749
    
    df_scale['Cholesterol'] = (df_scale['Cholesterol'] - 244.637939) / 53.385298

    if df_scale.loc[0, 'FastingBS'] == '> 120 mg/dL':
            df_scale.loc[0, 'FastingBS'] = 1
    elif df_scale.loc[0, 'FastingBS'] == 'Otherwise':
            df_scale.loc[0, 'FastingBS'] = 0
    
    if df_scale.loc[0, 'RestingECG'] == 'Left Ventricular Hypertrophy':
            df_scale.loc[0, 'RestingECG'] = 0
    elif df_scale.loc[0, 'RestingECG'] == 'Normal':
            df_scale.loc[0, 'RestingECG'] = 1   
    elif df_scale.loc[0, 'RestingECG'] == 'ST-T Wave Abnormality':
            df_scale.loc[0, 'RestingECG'] = 2
    
    df_scale['MaxHR'] = (df_scale['MaxHR'] - 136.789531) / 25.467129
    
    if df_scale.loc[0, 'ExerciseAngina'] == 'Yes':
            df_scale.loc[0, 'ExerciseAngina'] = 1
    elif df_scale.loc[0, 'ExerciseAngina'] == 'No':
            df_scale.loc[0, 'ExerciseAngina'] = 0
            
    # Oldpeak is normalised by min-maxing values following the prepared dataset
    df_scale['Oldpeak'] = (df_scale['Oldpeak'] - (-2.6)) / (6.2 - (-2.6))
    
    if df_scale.loc[0, 'ST_Slope'] == 'Downsloping':
            df_scale.loc[0, 'ST_Slope'] = 0
    elif df_scale.loc[0, 'ST_Slope'] == 'Flat':
            df_scale.loc[0, 'ST_Slope'] = 1   
    elif df_scale.loc[0, 'ST_Slope'] == 'Upsloping':
            df_scale.loc[0, 'ST_Slope'] = 2        

# Call function
scale_dataframe()

# Make predictions on the input array
prediction = RFC_model.predict(df_scale)
prediction_proba = RFC_model.predict_proba(df_scale)

# Tabs of different information
tab1, tab2, tab3 = st.tabs(["Dictionary", "Parameters", "Prediction"])

# Tab of dataset and dictionary
with tab1:
    
    # Original dataset before preparation
    st.subheader("Original Dataset from Kaggle")
    st.write(heart_bfmodel)
    
    st.subheader("Data Dictionary")

    st.write('''
    **Age:** The age of the patient in years.

    **Sex:** Biological sex of the patient.

    - **M:** Male

    - **F:** Female

    **ChestPainType:** Type of chest pain experienced.

    - **TA:** Typical Angina

    - **ATA:** Atypical Angina

    - **NAP:** Non-Anginal Pain

    - **ASY:** Asymptomatic

    **RestingBP:** Resting blood pressure in mm Hg.

    **Cholesterol:** Serum/Total cholesterol level in mm/dl.

    **FastingBS:** Fasting blood sugar of patient.

    - **1:** If FastingBS > 120 mg/dL

    - **0:** Otherwise

    **RestingECG:** Resting electrocardiogram results.

    - **Normal:** Normal

    - **ST:** Having ST-T Wave Abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)

    - **LVH:** Showing probable or definite Left Ventricular Hypertrophy by Estes' criteria

    **MaxHR:** Maximum heart rate achieved.

    **ExerciseAngina:** Presence of exercise-induced angina.

    - **Y:** Yes

    - **N:** No

    **Oldpeak:** ST numeric value measured in depression.

    **ST_Slope:** The slope of the peak exercise ST segment.

    - **Up:** Upsloping

    - **Flat:** Flat

    - **Down:** Downsloping

    **HeartDisease:** Output class of whether if patient has heart disease.

    - **1:** Heart Disease

    - **0:** Normal
        ''')
    
    # Dataset after data preparation and being used for modelling
    st.subheader("Dataset for Modelling after Data Preparation")
    st.write(heart)
    
    st.subheader("Data Dictionary")

    st.write('''
    **Age:** The age of the patient in years after stardardisation.

    **Sex:** Biological sex of the patient.

    - **0:** Female

    - **1:** Male

    **ChestPainType:** Type of chest pain experienced.
    
    - **0:** Asymptomatic

    - **1:** Atypical Angina

    - **2:** Non-Anginal Pain

    - **3:** Typical Angina

    **RestingBP:** Resting blood pressure in mm Hg after stardardisation.

    **Cholesterol:** Serum/Total cholesterol level in mm/dl after stardardisation.

    **FastingBS:** Fasting blood sugar of patient.

    - **0:** Otherwise

    - **1:** If FastingBS > 120 mg/dL

    **RestingECG:** Resting electrocardiogram results.

    - **0:** Showing probable or definite Left Ventricular Hypertrophy by Estes' criteria
    
    - **1:** Normal

    - **2:** Having ST-T Wave Abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)    

    **MaxHR:** Maximum heart rate achieved after stardardisation.

    **ExerciseAngina:** Presence of exercise-induced angina.

    - **0:** No

    - **1:** Yes

    **Oldpeak:** ST numeric value measured in depression after normalisation.

    **ST_Slope:** The slope of the peak exercise ST segment.

    - **0:** Downsloping

    - **1:** Flat

    - **2:** Upsloping

    **HeartDisease:** Output class of whether if patient has heart disease.

    - **0:** Normal

    - **1:** Heart Disease
        ''')

# Tab with original inputs and encoded + scaled inputs
with tab2:
    
    # The inputs from the sidebar inside a DataFrame
    st.subheader('User Input Parameters')
    st.write('The DataFrame below shows the parameters inputted.')
    st.write(df)
    
    # The inputs DataFrame encoded and scaled to match the prediction model and dataset
    st.subheader('Parameters after Encoding and Scaling')
    st.write('The DataFrame below shows the parameters inputted after **feature scaling** and **label encoding**.')
    st.write(df_scale)

# Tab with predicted value and probability of prediction
with tab3:
    
    # Explaination of the different values predicted
    st.subheader('Class Labels and their Corresponding Index Number')
    st.write(heart['HeartDisease'].unique())
    
    st.write('''
    **0:** This value means that the person does not have Coronary Heart Disease.
    
    **1:** This value means that the person does have Coronary Heart Disease or has signs of it.
    ''')

    # Predicted value of the inputs provided
    st.subheader('Predicted Value')
    st.write('Prediction:', heart['HeartDisease'].unique()[prediction])

    if prediction == 0:
        st.write('''
        The information predicts that the person is **not likely** to have Coronary Heart Disease.
        ''')
    else:
        st.write('''
        The information predicts that the person is **likely** to have or has Coronary Heart Disease.
        ''')
    
    # Prediction probability of the inputs and its likelihood for the person to have heart disease
    st.subheader('Prediction Probability')
    st.write(prediction_proba)
    
    prediction_proba_percent = prediction_proba * 100
    prediction_proba_percent = np.round(prediction_proba_percent, 1)
    
    st.write(
            f'''Probability of having heart disease: **{prediction_proba_percent[0, 1]}%**'''
        )
    
    # Prints a statement of the likelihood (low, high or unable to tell) of if the person would get/already has heart disease
    if (prediction_proba[0, 1] >= 0).any() and (prediction_proba[0, 1] <= 0.2).any():
        st.write('''
        The probability indicates a **very low likelihood** of having Coronary Heart Disease or signs of it.
        ''')
    elif (prediction_proba[0, 1] > 0.2).any() and (prediction_proba[0, 1] <= 0.4).any():
        st.write('''
        The probability indicates a **low likelihood** of having Coronary Heart Disease or signs of it.
        ''')
    elif (prediction_proba[0, 1] > 0.4).any() and (prediction_proba[0, 1] <= 0.6).any():
        st.write('''
        The probability **cannot tell** if the person has Coronary Heart Disease or signs of it.
        ''')
    elif (prediction_proba[0, 1] > 0.6).any() and (prediction_proba[0, 1] <= 0.8).any():
        st.write('''
        The probability indicates a **high likelihood** of having Coronary Heart Disease or signs of it.
        ''')
    else:
        st.write('''
        The probability indicates a **very high likelihood** of having Coronary Heart Disease or signs of it.
        ''')
        
# In[ ]:
    



