import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open('linear_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit app title
st.markdown("<h1 style='text-align: center;'>Restaurant Tip Predictor</h1>", unsafe_allow_html=True)

# User inputs
st.markdown("<h2 style='text-align: center;'>Enter the details below:</h2>", unsafe_allow_html=True)
total_bill = st.number_input("Total Bill ($)", min_value=0.0, format="%.2f", label_visibility="hidden")
size = st.number_input("Party Size", min_value=1, format="%d", label_visibility="hidden")

# Optional interaction term and squared features
total_bill_size = total_bill * size
total_bill_squared = total_bill ** 2
size_squared = size ** 2

# Prepare the feature vector
features = np.array([[total_bill, size, total_bill_size, total_bill_squared, size_squared]])

# Predict the tip
if st.button("Predict Tip"):
    predicted_tip = model.predict(features)[0]
    st.markdown(f"<h2 style='text-align: center;'>Predicted Tip: ${predicted_tip:.2f}</h2>", unsafe_allow_html=True)

