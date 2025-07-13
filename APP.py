
import streamlit as st
import numpy as np
from LSTM_MODEL import fetch_stock_data, train_model, predict_next_day
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly_graph import plot_realistic_forecast
import datetime

today = datetime.datetime.now().date()
if today.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
    st.warning("The stock market is closed on Saturdays and Sundays. No new data or predictions are available.")
else:
    # Place your forecasting logic here (fetch data, train, predict, plot)
    ...

st.title('Professional Stock Price Forecaster')

ticker = st.text_input('Enter Stock Ticker (e.g., AAPL):')

if st.button('Forecast'):
    if ticker:
        with st.spinner('Training model and generating forecast...'):
            try:
                data = fetch_stock_data(ticker)
                seq_length = 60
                model, scaler = train_model(data, epochs=10, seq_length=seq_length)
                forecast = predict_next_day(model, scaler, data, seq_length)
                st.success(f'Next day forecast for {ticker}: ${forecast:.2f}')
                fig = plot_realistic_forecast(ticker, data, forecast)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.error('Please enter a valid stock ticker.')
