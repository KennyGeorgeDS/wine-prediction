import streamlit as st
import requests

# FastAPI endpoint URL
API_URL = "http://acd7184e54d314c67844b5cba4797e3e-683884404.us-east-2.elb.amazonaws.com/predict"

# Streamlit app
st.title("Wine Prediction Dashboard")

st.sidebar.header("Enter Wine Features")

# Create input fields for all WineInput fields
fixed_acidity = st.sidebar.number_input("Fixed Acidity", min_value=0.0, max_value=20.0, value=7.4)
volatile_acidity = st.sidebar.number_input("Volatile Acidity", min_value=0.0, max_value=2.0, value=0.7)
citric_acid = st.sidebar.number_input("Citric Acid", min_value=0.0, max_value=2.0, value=0.0)
residual_sugar = st.sidebar.number_input("Residual Sugar", min_value=0.0, max_value=100.0, value=1.9)
chlorides = st.sidebar.number_input("Chlorides", min_value=0.0, max_value=1.0, value=0.076)
free_sulfur_dioxide = st.sidebar.number_input("Free Sulfur Dioxide", min_value=0.0, max_value=100.0, value=11.0)
total_sulfur_dioxide = st.sidebar.number_input("Total Sulfur Dioxide", min_value=0.0, max_value=400.0, value=34.0)
density = st.sidebar.number_input("Density", min_value=0.0, max_value=2.0, value=0.9978)
pH = st.sidebar.number_input("pH", min_value=0.0, max_value=14.0, value=3.51)
sulphates = st.sidebar.number_input("Sulphates", min_value=0.0, max_value=2.0, value=0.56)
alcohol = st.sidebar.number_input("Alcohol", min_value=0.0, max_value=20.0, value=9.4)

# Trigger prediction when the button is clicked
if st.sidebar.button("Predict Wine Type"):
    # Prepare the payload
    payload = {
        "fixed_acidity": fixed_acidity,
        "volatile_acidity": volatile_acidity,
        "citric_acid": citric_acid,
        "residual_sugar": residual_sugar,
        "chlorides": chlorides,
        "free_sulfur_dioxide": free_sulfur_dioxide,
        "total_sulfur_dioxide": total_sulfur_dioxide,
        "density": density,
        "pH": pH,
        "sulphates": sulphates,
        "alcohol": alcohol
    }

    # Query the FastAPI endpoint
    with st.spinner("Querying the model..."):
        response = requests.post(API_URL, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            st.success("Prediction Complete!")
            st.write(f"**Predicted Type:** {result['predicted_type']}")
            st.write(f"**Red Wine Probability:** {result['red_probability']:.4f}")
            st.write(f"**White Wine Probability:** {result['white_probability']:.4f}")
        else:
            st.error(f"Error: {response.status_code} - {response.text}")

