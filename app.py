import streamlit as st
import numpy as np
import joblib

# Initialize session state variable
if 'input_data' not in st.session_state:
    st.session_state['input_data'] = {}

def app_sidebar():
    st.sidebar.header('Housing Price Index Predictor')
    # User input in the sidebar
    quarter_num = st.sidebar.number_input('Enter the quarter number', value=1, min_value=1)

    def get_input_data():
        return {'quarter_num': quarter_num}

    predict_button = st.sidebar.button("Predict")
    reset_button = st.sidebar.button("Reset")

    if predict_button:
        st.session_state['input_data'] = get_input_data()
    if reset_button:
        st.session_state['input_data'] = {}

def app_body():
    st.title('Housing Price Index Predictor')
    # Display the prediction result in the main body
    if st.session_state['input_data']:
        model = joblib.load('housing_price_model.pkl')
        quarter_num = st.session_state['input_data']['quarter_num']
        prediction = model.predict(np.array([[quarter_num]]))
        st.write(f'Predicted Housing Price Index: {prediction[0]}')

def main():
    app_sidebar()
    app_body()

if __name__ == "__main__":
    main()
