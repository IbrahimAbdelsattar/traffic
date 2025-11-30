import sys
import subprocess

# --- Try to install xgboost if it's not already installed ---
def ensure_xgboost_installed():
    try:
        import xgboost  # just to check if it exists
    except ImportError:
        print("xgboost not found. Installing xgboost...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
        print("xgboost installed successfully.")

ensure_xgboost_installed()

from xgboost import XGBClassifier
import streamlit as st
import pickle
import pandas as pd
import numpy as np


# --- 1. Load Model and Encoders ---
# Ensure these files are in the same directory as your Streamlit app
with open('encoders.pkl', 'rb') as f:
    loaded_label_encoders = pickle.load(f)

loaded_xgboost_model = XGBClassifier()
loaded_xgboost_model.load_model('xgboost_accident_model.json')

# نفترض إنك درّبت الموديل على الأعمدة بعد ما اتمدوا بـ LabelEncoder فقط
# يبقى الـ feature names هي نفسها keys بتوع الـ encoders
model_features = list(loaded_label_encoders.keys())


# --- 2. Define Preprocessing Function ---
def preprocess_inputs(raw_data, label_encoders, model_features):
    """
    raw_data: dict of raw categorical inputs (strings)
    label_encoders: dict{column_name: fitted LabelEncoder}
    model_features: list of feature names used during training (same order)
    """
    # نشتغل على نسخة من الداتا كـ DataFrame
    processed_df = pd.DataFrame([raw_data])

    for col, encoder in label_encoders.items():
        if col in processed_df.columns:
            value_to_encode = processed_df[col].iloc[0]
            if value_to_encode in encoder.classes_:
                # نستخدم int() بدل astype على scalar
                encoded_val = int(encoder.transform([value_to_encode])[0])
                processed_df[col] = encoded_val
            else:
                # قيمة مش موجودة في التدريب → نخليها 0 أو أي default مناسب
                st.warning(
                    f"Unseen value '{value_to_encode}' for column '{col}'. Assigning 0."
                )
                processed_df[col] = 0

    # نتأكد إن كل الأعمدة رقمية
    processed_df = processed_df.astype(int)

    # نضمن إن الأعمدة مرتّبة زي ما الموديل اتدرّب عليها
    final_df = processed_df.reindex(columns=model_features)

    return final_df


# --- 3. Define Prediction Function ---
def predict_severity(
    age_band_of_driver,
    sex_of_driver,
    educational_level,
    vehicle_driver_relation,
    driving_experience,
    lanes_or_medians,
    types_of_junction,
    road_surface_type,
    light_conditions,
    weather_conditions,
    type_of_collision,
    vehicle_movement,
    pedestrian_movement,
    cause_of_accident
):
    raw_input_data = {
        'Age_band_of_driver': age_band_of_driver,
        'Sex_of_driver': sex_of_driver,
        'Educational_level': educational_level,
        'Vehicle_driver_relation': vehicle_driver_relation,
        'Driving_experience': driving_experience,
        'Lanes_or_Medians': lanes_or_medians,
        'Types_of_Junction': types_of_junction,
        'Road_surface_type': road_surface_type,
        'Light_conditions': light_conditions,
        'Weather_conditions': weather_conditions,
        'Type_of_collision': type_of_collision,
        'Vehicle_movement': vehicle_movement,
        'Pedestrian_movement': pedestrian_movement,
        'Cause_of_accident': cause_of_accident
    }

    preprocessed_input = preprocess_inputs(
        raw_input_data, loaded_label_encoders, model_features
    )

    prediction_numerical = loaded_xgboost_model.predict(preprocessed_input)[0]
    severity_mapping = {0: 'Fatal injury', 1: 'Serious injury', 2: 'Slight injury'}
    prediction_label = severity_mapping.get(
        int(prediction_numerical), 'Unknown Severity'
    )
    return prediction_label


# Helper to get categories for Streamlit selectbox
def get_categories_from_encoder(encoder_key):
    if encoder_key in loaded_label_encoders:
        return loaded_label_encoders[encoder_key].classes_.tolist()
    return []


# --- 4. Create Streamlit Interface ---
st.set_page_config(
    page_title="Road Traffic Accident Severity Prediction",
    layout="centered"
)

st.title("Road Traffic Accident Severity Prediction")
st.write("Enter the accident details to predict the severity.")

# Streamlit Input Widgets
with st.form("prediction_form"):
    age_band_of_driver = st.selectbox(
        "Age band of driver", get_categories_from_encoder('Age_band_of_driver')
    )
    sex_of_driver = st.selectbox(
        "Sex of driver", get_categories_from_encoder('Sex_of_driver')
    )
    educational_level = st.selectbox(
        "Educational level", get_categories_from_encoder('Educational_level')
    )
    vehicle_driver_relation = st.selectbox(
        "Vehicle driver relation", get_categories_from_encoder('Vehicle_driver_relation')
    )
    driving_experience = st.selectbox(
        "Driving experience", get_categories_from_encoder('Driving_experience')
    )
    lanes_or_medians = st.selectbox(
        "Lanes or Medians", get_categories_from_encoder('Lanes_or_Medians')
    )
    types_of_junction = st.selectbox(
        "Types of Junction", get_categories_from_encoder('Types_of_Junction')
    )
    road_surface_type = st.selectbox(
        "Road surface type", get_categories_from_encoder('Road_surface_type')
    )
    light_conditions = st.selectbox(
        "Light conditions", get_categories_from_encoder('Light_conditions')
    )
    weather_conditions = st.selectbox(
        "Weather conditions", get_categories_from_encoder('Weather_conditions')
    )
    type_of_collision = st.selectbox(
        "Type of collision", get_categories_from_encoder('Type_of_collision')
    )
    vehicle_movement = st.selectbox(
        "Vehicle movement", get_categories_from_encoder('Vehicle_movement')
    )
    pedestrian_movement = st.selectbox(
        "Pedestrian movement", get_categories_from_encoder('Pedestrian_movement')
    )
    cause_of_accident = st.selectbox(
        "Cause of accident", get_categories_from_encoder('Cause_of_accident')
    )

    submitted = st.form_submit_button("Predict Accident Severity")

    if submitted:
        prediction = predict_severity(
            age_band_of_driver,
            sex_of_driver,
            educational_level,
            vehicle_driver_relation,
            driving_experience,
            lanes_or_medians,
            types_of_junction,
            road_surface_type,
            light_conditions,
            weather_conditions,
            type_of_collision,
            vehicle_movement,
            pedestrian_movement,
            cause_of_accident
        )
        st.success(f"Predicted Accident Severity: **{prediction}**")
