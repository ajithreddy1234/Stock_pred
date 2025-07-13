import torch
import torch.nn as nn
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

def fetch_stock_data(ticker, period='5y'):
    data = yf.download(ticker, period=period)
    return data['Close'].values.reshape(-1, 1)

def preprocess_data(data, seq_length=60):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(data)
    x, y = [], []
    for i in range(seq_length, len(scaled_data)):
        x.append(scaled_data[i-seq_length:i, 0])
        y.append(scaled_data[i, 0])
    x = np.array(x)
    y = np.array(y)
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    return x, y, scaler

def train_model(data, epochs=10, seq_length=60):
    x, y, scaler = preprocess_data(data, seq_length)
    model = LSTMModel()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        for seq, labels in zip(x, y):
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))
            y_pred = model(seq)
            single_loss = loss_function(y_pred, labels.view(-1))
            single_loss.backward()
            optimizer.step()
    return model, scaler

def predict_next_day(model, scaler, data, seq_length=60):
    last_sequence = data[-seq_length:].reshape(-1, 1)
    scaled_last = scaler.transform(last_sequence)
    scaled_last = torch.from_numpy(scaled_last).float()
    model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                         torch.zeros(1, 1, model.hidden_layer_size))
    with torch.no_grad():
        pred = model(scaled_last[0])
    pred = scaler.inverse_transform(pred.detach().numpy().reshape(-1, 1))
    return pred[0][0]
