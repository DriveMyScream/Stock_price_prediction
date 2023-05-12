import tensorflow as tf
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

st.title("Stock Price Prediction Application")

# Dictionary to map stock names to model files
stock_models = {
    'Google': 'Google_Stock_Price_Prediction',
    'Microsoft': 'Microsoft_Stock_Price_Prediction',
    'Apple': 'Apple_Stock_Price_Prediction'
}

# Stock selection dropdown
selected_stock = st.selectbox('Select a stock', ('Google', 'Microsoft', 'Apple'))

# Load the selected stock model
model = load_model(stock_models[selected_stock])

# Price inputs
price_inputs = []
for i in range(1, 8):
    price_input = st.number_input(f'Enter the price for day {i}')
    price_inputs.append(price_input)

# Reshape the price inputs for prediction
prices = np.array(price_inputs).reshape(1, -1, 1)

# Prediction button
if st.button('Predict Price'):
    prediction_price = model.predict(prices)
    prediction_price = prediction_price[0][0]

    st.title(f"Predicted {selected_stock} Stock Price: {prediction_price:.2f}")
