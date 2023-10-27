import streamlit as st
import streamlit.components.v1 as components
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import pandas as pd
import pickle
from shap import Explainer, Explanation
import shap

st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(page_title="Heart Disease Detector", # page title, displayed on the window/tab bar
        		   page_icon="apple", # favicon: icon that shows on the window/tab bar (tip: you can use emojis)
                   layout="wide", # use full width of the page
                   menu_items={
                       'About': "App using various models to detect fruits and vegetables"
                   })

st.markdown("<h1 style='text-align: center; color: purple;'>Guardians of the Heartbeat</h1>", unsafe_allow_html=True)

left_co, cent_co, last_co = st.columns(3)

with cent_co:
    st.image("img/hero.jpg")

#-----
# Functions

def chest_pain_choice(choice):
    mapping = {
        'typical angina': 0,
        'atypical angina': 1,
        'non-anginal pain': 2,
        'asymptomatic': 3
    }
    return mapping.get(choice, "Invalid choice")


def resting_ecg_choice(choice):
    mapping = {
        'normal':0, 
        'ST-T wave abnormality T wave inversions and/or ST elevation or depression of > 0.05 mV':1,
        'left ventricular hypertrophy':2
    }
    return mapping.get(choice, "Invalid choice")


def st_slope_choice(choice):
    mapping = {
        'upsloping':1, 
        'flat':2,
        'downsloping':3
    }
    return mapping.get(choice, "Invalid choice")


def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)
#-----

#Inputs by user
age = st.number_input("Patient's Age: ")
sex = st.radio("Patient's Sex: ", ('Female', 'Male'), horizontal=True)
if sex == 'Female':
    sex = 0
else:
    sex = 1

chest_pain_type = st.selectbox("What type of chest pain? ", ('typical angina', 
                                                             'atypical angina', 
                                                             'non-anginal pain', 
                                                             'asymptomatic'))
resting_bp_s = st.slider("Resting Blood Pressure in mm Hg", min_value=0, max_value=200, value=120)
cholesterol = st.slider("Cholesterol Level in mg/dl", min_value=0, max_value=650, value=200)
fasting_blood_sugar = st.radio("Fasting Blood Sugar in mg/dl : ", ('Higher than 120', 'Lower than 120'), horizontal=True)
if fasting_blood_sugar == 'Higher than 120':
    fasting_blood_sugar = 1
else:
    fasting_blood_sugar = 0

resting_ecg = st.selectbox("Resting Electrocardiogram Results: ", ('normal', 
                                                                   'ST-T wave abnormality T wave inversions and/or ST elevation or depression of > 0.05 mV',
                                                                   'left ventricular hypertrophy'))
max_heart_rate = st.number_input("Maximum Heart Rate", min_value=60, max_value=220, value=60)
exercise_angina = st.radio("Does the patient have exercise angina? ", ('Yes', 'No'), horizontal=True)
if exercise_angina == 'Yes':
    exercise_angina = 1
else:
    exercise_angina = 0

oldpeak = st.number_input("ST depression induced by exercise: ", min_value=-3, max_value=7, value=0)
ST_slope = st.selectbox("ST Slope: ", ('upsloping', 
                                       'flat',
                                       'downsloping'))

features_dict = {'age':[int(age)], 'sex': [sex], 'chest_pain_type':[chest_pain_choice(chest_pain_type)], 
                 'resting_bp_s': [resting_bp_s], 'cholesterol': [cholesterol],
                 'fasting_blood_sugar': [fasting_blood_sugar], 'resting_ecg': [resting_ecg_choice(resting_ecg)], 
                 'max_heart_rate': [max_heart_rate], 'exercise_angina': [exercise_angina], 
                 'oldpeak': [oldpeak], 'ST_slope': [st_slope_choice(ST_slope)]}

df = pd.DataFrame.from_dict(features_dict)

button = st.button("Submit")

loaded_rf_model = pickle.load(open('model/random_forest_model', 'rb'))

if button:
    predictions = loaded_rf_model.predict(df)
    probabilities = loaded_rf_model.predict_proba(df)

    if predictions[0] == 0:
        left_co2, cent_co2, last_co2 = st.columns(3)
        with cent_co2:
            st.write('Hurray no heart disease')
            st.image("img/healthy-heart.jpg")
    else:
        left_co2, cent_co2, last_co2 = st.columns(3)
        with cent_co2:
            st.write(f'BE CAREFUL! RISK OF HEART DISEASE with {int(probabilities[0][1] * 100)}% probability')
            st.image("img/unhealthy-heart.jpg")

    best_model_explainer = Explainer(loaded_rf_model)
    shap_v2 = best_model_explainer(df)
    shap_exp = Explanation(shap_v2[:,:,1], shap_v2.base_values[:,1], df, feature_names=None)
    idx = 0 # datapoint to explain

    features_map = {'age':['feature 0'], 'sex': ['feature 1'], 'chest_pain_type':['feature 2'], 
                 'resting_bp_s': ['feature 3'], 'cholesterol': ['feature 4'],
                 'fasting_blood_sugar': ['feature 5'], 'resting_ecg': ['feature 6'], 
                 'max_heart_rate': ['feature 7'], 'exercise_angina': ['feature 8'], 
                 'oldpeak': ['feature 9'], 'ST_slope': ['feature 10']}

    st.write('Below are the highest risk factors that lead to this prediction')
    features = pd.DataFrame.from_dict(features_map)
    st.table(features)
    st.pyplot(shap.plots.waterfall(shap_exp[idx]))
