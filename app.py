import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load('/content/housing_price_model.pkl')

# Streamlit app title
st.title('Housing Price Index Predictor')

# User input
quarter_num = st.number_input('Enter the quarter number', value=1)

# Predict button
if st.button('Predict'):
    prediction = model.predict(np.array([[quarter_num]]))
    st.write(f'Predicted Housing Price Index: {prediction[0]}')
