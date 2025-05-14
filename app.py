import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("model.pkl")

st.set_page_config(page_title="Boston House Price Predictor", layout="centered")
st.title("🏠 House Price Predictor")
st.sidebar.markdown("👨‍💻 Made by Dev Patel")
st.write("Enter the features of a house to predict its price (in $1000s):")

# Input fields for the features
CRIM = st.number_input("CRIM: per capita crime rate by town", min_value=0.0)
ZN = st.number_input("ZN: proportion of residential land zoned over 25,000 sq.ft.", min_value=0.0)
INDUS = st.number_input("INDUS: proportion of non-retail business acres per town", min_value=0.0)
CHAS = st.selectbox("CHAS: bounds Charles River? (1 = Yes, 0 = No)", [0, 1])
RM = st.number_input("RM: average number of rooms per dwelling", min_value=0.0)
AGE = st.number_input("AGE: proportion of owner-occupied units built prior to 1940", min_value=0.0)
DIS = st.number_input("DIS: weighted distances to five Boston employment centres", min_value=0.0)
RAD = st.number_input("RAD: index of accessibility to radial highways", min_value=1.0)
TAX = st.number_input("TAX: full-value property-tax rate per $10,000", min_value=0.0)
PTRATIO = st.number_input("PTRATIO: pupil-teacher ratio by town", min_value=0.0)
B = st.number_input("B: 1000(Bk - 0.63)^2", min_value=0.0)
LSTAT = st.number_input("LSTAT: % lower status of the population", min_value=0.0)

# Prediction
if st.button("Predict House Price"):
    # Form the input vector as a 2D array
    input_features = np.array([[CRIM, ZN, INDUS, CHAS, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]])
    
    # Predict using the model
    prediction = model.predict(input_features)

    # Display result
    st.success(f"🏡 Predicted House Price: ${prediction[0] * 1000:,.2f}")
