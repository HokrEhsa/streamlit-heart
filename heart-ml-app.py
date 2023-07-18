#!/usr/bin/env python
# coding: utf-8

# In[1]:

# import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import streamlit as st

# Add Streamlit components
st.write('''
# Heart Disease Prediction App
This application will predict the possibility of having indicators of **Coronary Heart Disease**.
''')

# set up the sidebar
st.sidebar.image('heartbeatwave.png', use_column_width=True)

st.sidebar.header("User Input Parameters")

# import datasets
heart_bfmodel = pd.read_csv("heart.csv")
heart = pd.read_csv("heart_le.csv")

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

df = user_input_features()

df_scale = df.copy(deep=True)

def scale_dataframe():
    df_scale['Age'] = (df_scale['Age'] - 53.509269) / 9.437636
    
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
            
    df_scale['Oldpeak'] = (df_scale['Oldpeak'] - (-2.6)) / (6.2 - (-2.6))
    
    if df_scale.loc[0, 'ST_Slope'] == 'Downsloping':
            df_scale.loc[0, 'ST_Slope'] = 0
    elif df_scale.loc[0, 'ST_Slope'] == 'Flat':
            df_scale.loc[0, 'ST_Slope'] = 1   
    elif df_scale.loc[0, 'ST_Slope'] == 'Upsloping':
            df_scale.loc[0, 'ST_Slope'] = 2                
            
scale_dataframe()

X = heart.drop('HeartDisease', axis=1)
y = heart['HeartDisease']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=2201984)

model = RandomForestClassifier(max_depth=5, criterion='gini', min_samples_split=2, n_estimators=300, max_features='sqrt')
model.fit(X_train, y_train)

# Make predictions on the input array
prediction = model.predict(df_scale)
prediction_proba = model.predict_proba(df_scale)



tab1, tab2, tab3 = st.tabs(["Dictionary", "Parameters", "Prediction"])

with tab1:
    st.subheader("Dataset used for Modelling")
    st.write(heart_bfmodel)
    
    st.subheader("Data Dictionary")
#     st.write('''
# **Age:** The age of the patient in years.

# **Sex:** Biological sex of the patient.

# **ChestPainType:** Type of chest pain experienced.

# **RestingBP:** Resting blood pressure in mm Hg.

# **Cholesterol:** Serum/Total cholesterol level in mm/dl.

# **FastingBS:** Fasting blood sugar of patient.

# **RestingECG:** Resting electrocardiogram results.

# **MaxHR:** Maximum heart rate achieved.

# **ExerciseAngina:** Presence of exercise-induced angina.

# **Oldpeak:** ST numeric value measured in depression.

# **ST_Slope:** The slope of the peak exercise ST segment.

# **HeartDisease:** Output class of whether if patient has heart disease.
# ''')

    st.write('''
    **Age:** The age of the patient in years.

    **Sex:** Biological sex of the patient.

    - M: Male

    - F: Female

    **ChestPainType:** Type of chest pain experienced.

    - TA: Typical Angina

    - ATA: Atypical Angina

    - NAP: Non-Anginal Pain

    - ASY: Asymptomatic

    **RestingBP:** Resting blood pressure in mm Hg.

    **Cholesterol:** Serum/Total cholesterol level in mm/dl.

    **FastingBS:** Fasting blood sugar of patient.

    - 1: If FastingBS > 120 mg/dL

    - 0: Otherwise

    **RestingECG:** Resting electrocardiogram results.

    - Normal: Normal

    - ST: Having ST-T Wave Abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)

    - LVH: Showing probable or definite Left Ventricular Hypertrophy by Estes' criteria

    **MaxHR:** Maximum heart rate achieved.

    **ExerciseAngina:** Presence of exercise-induced angina.

    - Y: Yes

    - N: No

    **Oldpeak:** ST numeric value measured in depression.

    **ST_Slope:** The slope of the peak exercise ST segment.

    - Up: Upsloping

    - Flat: Flat

    - Down: Downsloping

    **HeartDisease:** Output class of whether if patient has heart disease.

    - 1: Heart Disease

    - 0: Normal
        ''')

    
with tab2:
    st.subheader('User Input Parameters')
    st.write('The DataFrame below shows the parameters inputted.')
    st.write(df)
    
    st.subheader('Parameters after Encoding and Scaling')
    st.write('The DataFrame below shows the parameters inputted after **feature scaling** and **label encoding**.')
    st.write(df_scale)
    
with tab3:
    st.subheader('Class Labels and their Corresponding Index Number')
    st.write(heart['HeartDisease'].unique())

    st.write('''
    **0:** This value means that the person does not have Coronary Heart Disease.
    
    **1:** This value means that the person does have Coronary Heart Disease or has signs of it.
    ''')

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

    st.subheader('Prediction Probability')
    st.write(prediction_proba)
    
#     st.write('''
#         1. If the probability of Class 1 is significantly higher than **0.5**, the parameters indicate that the person has a **high likelihood** of having heart disease.
#         ''')
#     st.write('''
#         2. If the probability of Class 1 is significantly lower than **0.5**, the parameters indicate that the person has a **low likelihood** of having heart disease.
#         ''')
#     st.write('''
#         3. If the probability of Class 1 is around **0.5** (or has a probability similar in value to Class 0), the parameters are **unable to indicate** if the person has a high or low likelihood of having heart disease.
#         ''')
    
    if (prediction_proba >= 0).any() and (prediction_proba <= 0.2).any():
        st.write('''
        The probability indicates a **very high likelihood** of having Coronary Heart Disease or signs of it.
        ''')
    elif (prediction_proba > 0.2).any() and (prediction_proba <= 0.4).any():
        st.write('''
        The probability indicates a **high likelihood** of having Coronary Heart Disease or signs of it.
        ''')
    elif (prediction_proba > 0.4).any() and (prediction_proba <= 0.6).any():
        st.write('''
        The probability is **unable to tell** if the person has Coronary Heart Disease or signs of it.
        ''')
    elif (prediction_proba > 0.6).any() and (prediction_proba <= 0.8).any():
        st.write('''
        The probability indicates a **low likelihood** of having Coronary Heart Disease or signs of it.
        ''')
    else:
        st.write('''
        The probability indicates a **very low likelihood** of having Coronary Heart Disease or signs of it.
        ''')


    
# In[ ]:




