import plotly.graph_objs as go
import pandas as pd

def plot_realistic_forecast(ticker, data, forecast):
    # Prepare DataFrame
    df = pd.DataFrame(data, columns=['Close'])
    df['Open'] = df['Close'].shift(1)
    df['Open'].iloc[0] = df['Close'].iloc[0]
    df['High'] = df[['Open', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'Close']].min(axis=1)
    df['Date'] = pd.date_range(end=pd.Timestamp.today(), periods=len(df))

    # Add forecast as the next point
    next_date = df['Date'].iloc[-1] + pd.Timedelta(days=1)
    forecast_row = pd.DataFrame({
        'Date': [next_date],
        'Open': [df['Close'].iloc[-1]],
        'High': [max(df['Close'].iloc[-1], forecast)],
        'Low': [min(df['Close'].iloc[-1], forecast)],
        'Close': [forecast]
    })
    df_all = pd.concat([df, forecast_row], ignore_index=True)

    # Candlestick for actual data
    fig = go.Figure(data=[go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        increasing_line_color='green',
        decreasing_line_color='red',
        name='Actual'
    )])

    # Overlay actual close prices as a line
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Close'],
        mode='lines+markers',
        name='Actual Close',
        line=dict(color='black', width=2)
    ))

    # Overlay prediction as a line from last actual to forecast
    fig.add_trace(go.Scatter(
        x=[df['Date'].iloc[-1], next_date],
        y=[df['Close'].iloc[-1], forecast],
        mode='lines+markers',
        name='Predicted',
        line=dict(color='blue', width=3, dash='dot'),
        marker=dict(size=10, color='blue')
    ))

    # Formatting
    fig.update_layout(
        title=f'{ticker} Stock Price: Actual and Forecast',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        template='plotly_white',
        width=1000,
        height=500
    )
    return fig
