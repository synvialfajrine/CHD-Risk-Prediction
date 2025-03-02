import streamlit as st
import pickle
import numpy as np

# Set the page title
st.set_page_config(page_title="CHD Risk Prediction")

# Define project path and model file
project_path = "C:\\Users\\USER\\Documents\\cardiovascular-heart-risk"
model_file_path = ("xgb_model.pkl")

# Load the trained model with caching
@st.cache_resource
def load_model():
    try:
        with open(model_file_path, "rb") as file:
            model = pickle.load(file)
        if not hasattr(model, "predict"):
            raise ValueError("Loaded model does not have a 'predict' method. Check the model file.")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load model
model = load_model()

def predict_risk(features):
    if model is None:
        return None  # Handle missing model case
    probability = model.predict_proba([features])[:, 1]  # Get probability of the positive class
    prediction = 1 if probability >= 0.35 else 0  # Apply threshold
    return prediction




def get_recommendations(risk):
    if risk == 1:
        return "You have a high risk of coronary heart disease. Consider consulting a doctor for a thorough evaluation, adopting a healthier diet, exercising regularly, and quitting smoking if applicable."
    else:
        return "Your risk of coronary heart disease is low. However, it is still advisable to maintain a healthy lifestyle and have regular check-ups with your doctor."

# Streamlit UI
st.title("Cardiovascular 10-Year Risk Prediction")
st.write("Fill in the details below to assess your 10-year risk of coronary heart disease (CHD).")

# Form for user input
with st.form("risk_form"):
    age = st.number_input("What is your age?", min_value=1, max_value=120, step=1, value=None)
    education = st.selectbox("What is your highest level of education?", [None, 4, 3, 2, 1], format_func=lambda x: "Select" if x is None else {4: "Tertiary School", 3: "High School", 2: "Middle School", 1: "Primary School"}.get(x, x))
    sex = st.selectbox("What is your sex?", [None, "Female", "Male"], format_func=lambda x: "Select" if x is None else x)
    is_smoking = st.selectbox("Do you currently smoke?", [None, "Yes", "No"], format_func=lambda x: "Select" if x is None else x)
    cigs_per_day = st.number_input("How many cigarettes do you smoke per day on average?", min_value=0, step=1, value=None)
    bp_meds = st.selectbox("Are you currently on blood pressure medication?", [None, "Yes", "No"], format_func=lambda x: "Select" if x is None else x)
    prevalent_stroke = st.selectbox("Have you ever had a stroke?", [None, "Yes", "No"], format_func=lambda x: "Select" if x is None else x)
    prevalent_hyp = st.selectbox("Do you have hypertension?", [None, "Yes", "No"], format_func=lambda x: "Select" if x is None else x)
    diabetes = st.selectbox("Do you have diabetes?", [None, "Yes", "No"], format_func=lambda x: "Select" if x is None else x)
    tot_chol = st.number_input("What is your total cholesterol level?", min_value=100.0, max_value=600.0, step=0.1, value=None)
    sys_bp = st.number_input("What is your systolic blood pressure?", min_value=80.0, max_value=250.0, step=0.1, value=None)
    dia_bp = st.number_input("What is your diastolic blood pressure?", min_value=50.0, max_value=150.0, step=0.1, value=None)
    bmi = st.number_input("What is your Body Mass Index (BMI)?", min_value=10.0, max_value=50.0, step=0.1, value=None)
    heart_rate = st.number_input("What is your heart rate?", min_value=40.0, max_value=200.0, step=0.1, value=None)
    glucose = st.number_input("What is your glucose level?", min_value=50.0, max_value=300.0, step=0.1, value=None)
    
    submit_button = st.form_submit_button("Predict Risk")

if submit_button:
    if model is None:
        st.error("Model could not be loaded. Please check the model file.")
    else:
        # Convert categorical inputs to numerical
        sex = 1 if sex == "Male" else 0 if sex == "Female" else None
        is_smoking = 1 if is_smoking == "Yes" else 0 if is_smoking == "No" else None
        bp_meds = 1 if bp_meds == "Yes" else 0 if bp_meds == "No" else None
        prevalent_stroke = 1 if prevalent_stroke == "Yes" else 0 if prevalent_stroke == "No" else None
        prevalent_hyp = 1 if prevalent_hyp == "Yes" else 0 if prevalent_hyp == "No" else None
        diabetes = 1 if diabetes == "Yes" else 0 if diabetes == "No" else None

        # Prepare feature array
        features = np.array([age, education, sex, is_smoking, cigs_per_day, bp_meds, prevalent_stroke, 
                            prevalent_hyp, diabetes, tot_chol, sys_bp, dia_bp, bmi, heart_rate, glucose])

        if None in features:
            st.error("Please fill in all the required fields before submitting.")
        else:
            # Make prediction
            risk = predict_risk(features)

            if risk is None:
                st.error("An error occurred during prediction.")
            else:
                recommendations = get_recommendations(risk)
                
                # Display results
                st.subheader("Prediction Result")
                st.write(f"Risk of CHD: **{'High' if risk == 1 else 'Low'}**")
                st.subheader("Recommendations")
                st.write(recommendations)

st.write("**Disclaimer:** This tool provides an estimation based on input data and should not replace professional medical evaluation. Please consult your doctor for a more reliable assessment.")