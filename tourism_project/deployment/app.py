import streamlit as st 
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the tourism model from Hugging Face Hub
model_path = hf_hub_download(
    repo_id="diwakarpandey-greatlearning/tripprediction-model",
    filename="best_tripprediction_model_v1.joblib"
)

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Prediction
st.title("Tourism Package Purchase Prediction App")
st.write("This app predicts whether a customer will purchase a tourism package based on their profile and interaction data.")

# Collect user input
Age = st.number_input("Age of Customer", min_value=18, max_value=100, value=30)
CityTier = st.selectbox("City Tier", [1, 2, 3], index=0)
DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=60, value=10)
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=2)
PreferredPropertyStar = st.selectbox("Preferred Property Star", [3, 4, 5], index=0)
NumberOfTrips = st.number_input("Number of Trips per Year", min_value=0, max_value=20, value=2)

# Passport and Car as Yes/No
Passport = st.selectbox("Passport", ["Yes", "No"], index=1)
OwnCar = st.selectbox("Own Car", ["Yes", "No"], index=1)

PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
MonthlyIncome = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=25000)

TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"], index=0)

# Expanded Occupation list
Occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Free Lancer", "Small Business", "Large Business"],
    index=0
)

Gender = st.selectbox("Gender", ["Male", "Female"], index=0)

# Expanded ProductPitched list
ProductPitched = st.selectbox(
    "Product Pitched",
    ["Deluxe", "Basic", "Standard", "King", "Super Deluxe"],
    index=0
)

MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"], index=0)
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"], index=0)

# Build input dataframe
input_data = pd.DataFrame([{
    "Age": Age,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "PreferredPropertyStar": PreferredPropertyStar,
    "NumberOfTrips": NumberOfTrips,
    "Passport": 1 if Passport == "Yes" else 0,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": 1 if OwnCar == "Yes" else 0,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "MonthlyIncome": MonthlyIncome,
    "TypeofContact": TypeofContact,
    "Occupation": Occupation,
    "Gender": Gender,
    "ProductPitched": ProductPitched,
    "MaritalStatus": MaritalStatus,
    "Designation": Designation
}])

# Prediction
classification_threshold = 0.45
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "purchase the package" if prediction == 1 else "not purchase the package"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
    st.write(f"Probability of purchase: {prediction_proba:.2f}")
