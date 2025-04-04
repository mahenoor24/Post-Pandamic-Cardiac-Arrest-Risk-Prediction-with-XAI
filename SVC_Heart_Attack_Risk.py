import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
import seaborn as sns


# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("heart_attack1.csv")
    return df

# Preprocess dataset
def preprocess_data(df):
    label_encoders = {}
    categorical_columns = [
        "Gender", "Pre_existing_Conditions", "Family_History", "Exercise_Habits",
        "Diet", "Smoking_Status", "Alcohol_Consumption", "COVID_Severity",
        "ICU_Admission", "Ventilator_Support", "Oxygen_Requirement",
        "Medications_Taken", "Post_COVID_Symptoms", "Cholesterol_Levels",
        "ECG_Results", "Echocardiogram_Results"
    ]

    for column in categorical_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    X = df.drop("Heart_Attack_Risk", axis=1)
    y = df["Heart_Attack_Risk"].map({"Low": 0, "Medium": 1, "High": 2})

    return X, y, label_encoders

# Train model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    feature_names = X.columns  # Store feature names before scaling

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = SVC(kernel='linear', probability=True, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, scaler, X_train, X_test, y_test, y_pred, accuracy, feature_names


# Define high-risk values for explanation
HIGH_RISK_FEATURES = {
    "ICU_Admission": "Yes",
    "Ventilator_Support": "Yes",
    "Oxygen_Requirement": "Yes",
    "Pre_existing_Conditions": ["Diabetes", "Hypertension", "Obesity", "High_Cholesterol", "Smoking", "Multiple"],
    "Exercise_Habits": "Sedentary",
    "Diet": "Poor",
    "Smoking_Status": "Yes",
    "Cholesterol_Levels": "High",
    "ECG_Results": "Abnormal",
    "Family_History": "Yes",
    "Echocardiogram_Results": "Abnormal"
}

def get_pfi_explanation(model, X_test, y_test, user_inputs, predicted_risk, feature_names):
    importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    feature_importances = sorted(
        zip(feature_names, importance.importances_mean),
        key=lambda x: x[1], 
        reverse=True
    )

    explanation = f"**The predicted heart attack risk is {predicted_risk} because:**\n\n"
    high_risk_factors = []
    
    for feature, _ in feature_importances:
        user_value = user_inputs.get(feature, "Unknown")
        
        if feature in HIGH_RISK_FEATURES:
            high_risk_values = HIGH_RISK_FEATURES[feature]
            if isinstance(high_risk_values, list):
                if user_value in high_risk_values:
                    high_risk_factors.append(f"- **{feature}** is '{user_value}', which is a **high-risk factor**.")
            else:
                if user_value == high_risk_values:
                    high_risk_factors.append(f"- **{feature}** is '{user_value}', which is a **high-risk factor**.")
    
    explanation += "\n".join(high_risk_factors) if high_risk_factors else "- No major high-risk factors detected."
    
    fig, ax = plt.subplots(figsize=(8, 5))
    features = [f[0] for f in feature_importances[:10]]
    scores = [f[1] for f in feature_importances[:10]]
    colors = ["red" if user_inputs.get(f[0], "Unknown") in HIGH_RISK_FEATURES.get(f[0], []) else "blue" for f in feature_importances[:10]]
    ax.barh(features[::-1], scores[::-1], color=colors[::-1])
    ax.set_xlabel("Feature Importance Score")
    ax.set_title("Key Factors Influencing Risk")
    ax.grid(axis="x", linestyle="--", alpha=0.7)

    return explanation, fig


def predict_risk(inputs, model, scaler, X_train, X_test, y_test, label_encoders, feature_names):
    encoded_inputs = []
    for column, value in inputs.items():
        if column in label_encoders:
            le = label_encoders[column]
            encoded_inputs.append(le.transform([value])[0] if value in le.classes_ else 0)
        else:
            encoded_inputs.append(value)

    encoded_inputs = np.array([encoded_inputs])  # Convert to NumPy array
    encoded_inputs = scaler.transform(encoded_inputs)  # Scale the inputs

    risk = model.predict(encoded_inputs)[0]
    risk_label = {0: "Low", 1: "Medium", 2: "High"}[risk]

    explanation, fig = get_pfi_explanation(model, X_test, y_test, inputs, risk_label, feature_names)
    return risk_label, explanation, fig

def generate_precautions(inputs, risk):
    precautions = []

    # General precautions based on risk level
    if risk == "High":
        precautions.append("🚨 Consult a cardiologist immediately for detailed assessment.")
        precautions.append("📊 Regular monitoring of blood pressure, cholesterol, and ECG is advised.")
    elif risk == "Medium":
        precautions.append("✅ Adopt a heart-healthy diet and increase physical activity.")
        precautions.append("📅 Schedule regular health checkups.")
    else:
        precautions.append("👍 Maintain a healthy lifestyle to keep your risk low.")

    # Specific precautions based on user inputs
    if inputs["Pre_existing_Conditions"] in ["Diabetes", "Hypertension", "Obesity", "High_Cholesterol", "Multiple"]:
        precautions.append("🩺 Manage pre-existing conditions through medication, diet, and regular follow-ups.")

    if inputs["Exercise_Habits"] == "Sedentary":
        precautions.append("🏃‍♂️ Gradually increase physical activity — aim for at least 30 minutes a day.")

    if inputs["Diet"] == "Poor":
        precautions.append("🥗 Improve your diet — consume more fruits, vegetables, and whole grains.")

    if inputs["Smoking_Status"] == "Yes":
        precautions.append("🚭 Quit smoking to significantly reduce your heart disease risk.")

    if inputs["Alcohol_Consumption"] == "Excessive":
        precautions.append("🍷 Limit alcohol intake — excessive alcohol increases heart risk.")

    if inputs["Cholesterol_Levels"] == "High":
        precautions.append("🧬 Focus on lowering cholesterol through diet, exercise, and medication if required.")

    if inputs["ECG_Results"] == "Abnormal":
        precautions.append("📉 Follow up with a cardiologist for further investigation and potential treatment.")

    if inputs["Echocardiogram_Results"] == "Abnormal":
        precautions.append("🫀 Further cardiac imaging or tests may be required to evaluate heart function.")

    if inputs["Post_COVID_Symptoms"] in ["Chest_Pain", "Breathlessness", "Palpitations", "Multiple"]:
        precautions.append("🏥 Seek post-COVID care for persistent symptoms affecting heart and lungs.")

    if inputs["ICU_Admission"] == "Yes" or inputs["Ventilator_Support"] == "Yes" or inputs["Oxygen_Requirement"] == "Yes":
        precautions.append("🏥 Post-hospital recovery should include regular heart and lung health assessments.")

    return precautions


def display_metrics(y_test, y_pred):
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=["Low", "Medium", "High"], columns=["Low", "Medium", "High"])
    
    # Precision, Recall, F1 Score
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    
    # Display confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    st.pyplot(fig)
    
    # Display Precision, Recall, F1 Score
    st.write(f"**Precision:** {precision:.2f}")
    st.write(f"**Recall:** {recall:.2f}")
    st.write(f"**F1 Score:** {f1:.2f}")


# Streamlit UI

st.title("Heart Attack Risk Prediction")

df = load_data()
X, y, label_encoders = preprocess_data(df)
model, scaler, X_train, X_test, y_test, y_pred, accuracy, feature_names = train_model(X, y)

st.write(f"**Support Vector Classifier Model Accuracy:** {accuracy:.2f}")

# User input fields
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    pre_existing_conditions = st.selectbox("Pre-existing Conditions", ["None", "Diabetes", "Hypertension", "Obesity", "High_Cholesterol", "Smoking", "Multiple"])
    family_history = st.selectbox("Family History", ["Yes", "No"])
    exercise_habits = st.selectbox("Exercise Habits", ["Sedentary", "Moderate", "Active"])
    diet = st.selectbox("Diet", ["Poor", "Average", "Healthy"])

with col2:
    smoking_status = st.selectbox("Smoking Status", ["Yes", "No"])
    alcohol_consumption = st.selectbox("Alcohol Consumption", ["None", "Moderate", "Excessive"])
    covid_severity = st.selectbox("COVID Severity", ["Mild", "Moderate", "Severe"])
    icu_admission = st.selectbox("ICU Admission", ["Yes", "No"])
    ventilator_support = st.selectbox("Ventilator Support", ["Yes", "No"])
    oxygen_requirement = st.selectbox("Oxygen Requirement", ["Yes", "No"])

with col3:
    medications_taken = st.selectbox("Medications Taken", ["None", "Steroids", "Antivirals", "Blood_Thinners", "Multiple"])
    hospitalization_duration = st.number_input("Hospitalization Duration (days)", min_value=0, max_value=30)
    post_covid_symptoms = st.selectbox("Post-COVID Symptoms", ["None", "Fatigue", "Chest_Pain", "Breathlessness", "Palpitations", "Multiple"])
    cholesterol_levels = st.selectbox("Cholesterol Levels", ["Normal", "Borderline", "High"])
    ecg_results = st.selectbox("ECG Results", ["Normal", "Abnormal"])
    echocardiogram_results = st.selectbox("Echocardiogram Results", ["Normal", "Abnormal"])

# Collect inputs
inputs = {
    "Age": age,
    "Gender": gender,
    "Pre_existing_Conditions": pre_existing_conditions,
    "Family_History": family_history,
    "Exercise_Habits": exercise_habits,
    "Diet": diet,
    "Smoking_Status": smoking_status,
    "Alcohol_Consumption": alcohol_consumption,
    "COVID_Severity": covid_severity,
    "ICU_Admission": icu_admission,
    "Ventilator_Support": ventilator_support,
    "Oxygen_Requirement": oxygen_requirement,
    "Medications_Taken": medications_taken,
    "Hospitalization_Duration": hospitalization_duration,
    "Post_COVID_Symptoms": post_covid_symptoms,
    "Cholesterol_Levels": cholesterol_levels,
    "ECG_Results": ecg_results,
    "Echocardiogram_Results": echocardiogram_results,
}


if st.button("Predict Risk"):
    risk, explanation, fig = predict_risk(inputs, model, scaler, X_train, X_test, y_test, label_encoders, feature_names)
    precautions = generate_precautions(inputs, risk)
    st.write(f"**Heart Attack Risk:** {risk}")
    st.write("### Explanation")
    st.write(explanation)
    st.pyplot(fig)

    st.write("### Precautionary Measures to Take")
    for precaution in precautions:
        st.write(f"- {precaution}")

    # Display confusion matrix and metrics
    display_metrics(y_test, y_pred)

