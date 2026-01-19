
import streamlit as st
import joblib
import pandas as pd

@st.cache_resource
def load_assets():
    model = joblib.load("ad_click_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_assets()

st.title("📢 Ad Click Prediction App")
st.write("Predict whether a user will click on an advertisement")

# ---------------- USER INPUTS ----------------
Daily_Time_Spent_on_Site = st.number_input("Daily Time Spent on Site", min_value=0.0)
Age = st.number_input("Age", min_value=1)
Area_Income = st.number_input("Area Income", min_value=0.0)
Daily_Internet_Usage = st.number_input("Daily Internet Usage", min_value=0.0)
Gender = st.selectbox("Gender", ["Male", "Female"])

# 🔹 Country dropdown generated from trained model
country_list = sorted(
    [c.replace("Country_", "") for c in scaler.feature_names_in_ if c.startswith("Country_")]
)
Country = st.selectbox("Country", country_list)

# ---------------- PREDICTION ----------------
if st.button("Predict"):

    gender_numeric = 1 if Gender == "Male" else 0

    # 1️⃣ Base features
    input_dict = {
        "Daily Time Spent on Site": Daily_Time_Spent_on_Site,
        "Age": Age,
        "Area Income": Area_Income,
        "Daily Internet Usage": Daily_Internet_Usage,
        "Gender": gender_numeric
    }

    # 2️⃣ Add all Country columns as 0
    for col in scaler.feature_names_in_:
        if col.startswith("Country_"):
            input_dict[col] = 0

    # 3️⃣ Set selected country to 1
    input_dict[f"Country_{Country}"] = 1

    # 4️⃣ Create DataFrame in EXACT training order
    input_data = pd.DataFrame(
        [input_dict],
        columns=scaler.feature_names_in_
    )

    # 5️⃣ Scale + predict
    input_scaled = scaler.transform(input_data)
    prob = model.predict_proba(input_scaled)[0][1]

    # ---------------- OUTPUT ----------------
    st.write(f"📊 Click Probability: **{prob * 100:.2f}%**")

    if prob >= 0.5:
        st.success("✅ User is likely to CLICK the ad")
    else:
        st.error("❌ User is NOT likely to click the ad")
