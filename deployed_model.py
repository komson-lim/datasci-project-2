# Import libs
import pandas as pd
import numpy as np

import torch
from torch import nn

import datetime


class PMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, device, drop_prob=0.2):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device

        self.gru = nn.GRU(
            input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob
        )
        self.fc = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = torch.zeros(size=(self.n_layers, len(x),
                        self.hidden_dim)).to(self.device)
        y, _ = self.gru(x, h)
        y = self.fc(y[:, -1])
        y = self.relu(y)
        return y


def initilize_model(device, station):
    input_dim = 5
    hidden_dim = 512
    n_layers = 5

    pm_model = PMModel(input_dim, hidden_dim, n_layers, device)
    pm_model = pm_model.to(device)

    best_weights_path = f"weights/{station}_best_weights.pth"
    pm_model.load_state_dict(torch.load(best_weights_path))

    return pm_model


def add_feature(df):
    df["Hourcos"] = np.cos(2 * np.pi * df.index.hour / 24)
    df["Hoursin"] = np.sin(2 * np.pi * df.index.hour / 24)

    df["Weekcos"] = np.cos(2 * np.pi * (df.index.isocalendar().week - 1) / 52)
    df["Weeksin"] = np.sin(2 * np.pi * (df.index.isocalendar().week - 1) / 52)
    return df


def process_input(df):
    # input is dataframe with Datetime and PM2.5
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    pred_time = df.iloc[-1, df.columns.get_loc("Datetime")] + datetime.timedelta(
        hours=1
    )
    df = df.set_index("Datetime")
    df = add_feature(df)
    return (np.expand_dims(df.to_numpy().astype(np.float32), axis=0), pred_time)


def predict(model, x, device):
    model.eval()
    with torch.no_grad():
        x = torch.Tensor(x).to(device)
        y_pred = model(x)
        y_pred = y_pred.detach().cpu().numpy()[0][0]
    return y_pred


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    station = "76t"
    pm_model = initilize_model(device, station)
    # Example input
    input_df = pd.read_csv("75t_prediction.csv").iloc[:24, :]
    x, pred_time = process_input(input_df)
    y_pred = predict(pm_model, x, device)
    print(f"Time: {pred_time} Prediction: {y_pred}")


if __name__ == "__main__":
    main()
