from datetime import datetime, timedelta, tzinfo
import json
import os
from airflow.models import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
import requests
import pandas as pd
import numpy as np
import torch
from torch import nn
import pytz
import logging
from google.cloud import storage

FETCH_URL = "http://air4thai.com/forweb/getHistoryData.php"


def fetch_history(station: str, pred_time: datetime):
    end = pred_time - timedelta(hours=1)
    end_day = end.day
    end_month = end.month
    end_year = end.year
    end_hour = end.hour

    start = pred_time - timedelta(days=1)
    start_day = start.day
    start_month = start.month
    start_year = start.year
    start_hour = start.hour

    params = {
        "stationID": station,
        "param": "PM25",
        "type": "hr",
        "sdate": f"{start_year}-{start_month}-{start_day}",
        "edate": f"{end_year}-{end_month}-{end_day}",
        "stime": f"{start_hour}",
        "etime": f"{end_hour}",
    }
    logging.info(params)

    data = requests.get(FETCH_URL, params=params).json()
    if data["result"] != "OK":
        raise Exception(f"Failed to fetch data {json.dumps(data)}")

    history = data["stations"][0]["data"]
    df_data = {
        "Datetime": [e["DATETIMEDATA"] for e in history],
        "PM2.5": [e["PM25"] for e in history],
    }

    df = pd.DataFrame.from_dict(df_data)
    df["PM2.5"] = df["PM2.5"].interpolate(limit_direction="both")
    logging.info(df)
    return df


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
        h = torch.zeros(size=(self.n_layers, len(x), self.hidden_dim)).to(self.device)
        y, _ = self.gru(x, h)
        y = self.fc(y[:, -1])
        y = self.relu(y)
        return y


CUR_DIR = os.path.abspath(os.path.dirname(__file__))


def initialize_model(device, station):
    input_dim = 5
    hidden_dim = 512
    n_layers = 5

    pm_model = PMModel(input_dim, hidden_dim, n_layers, device)
    pm_model = pm_model.to(device)

    best_weights_path = f"{CUR_DIR}/weights/{station}_best_weights.pth"
    pm_model.load_state_dict(
        torch.load(best_weights_path, map_location=torch.device("cpu"))
    )

    return pm_model


def add_feature(df):
    df["Hourcos"] = np.cos(2 * np.pi * df.index.hour / 24)
    df["Hoursin"] = np.sin(2 * np.pi * df.index.hour / 24)

    df["Weekcos"] = np.cos(2 * np.pi * (df.index.isocalendar().week - 1) / 52)
    df["Weeksin"] = np.sin(2 * np.pi * (df.index.isocalendar().week - 1) / 52)
    return df


def process_input(df) -> tuple[np.ndarray, pd.Timestamp]:
    # input is dataframe with Datetime and PM2.5
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    pred_time = df.iloc[-1, df.columns.get_loc("Datetime")] + timedelta(hours=1)
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


aqi_levels = [
    "Very Good",
    "Good",
    "Moderate",
    "Unhealthy for Sensitive Groups",
    "Unhealthy",
]


def get_AQI(x):
    if x < 25:
        return aqi_levels[0]
    elif x < 37:
        return aqi_levels[1]
    elif x < 50:
        return aqi_levels[2]
    elif x < 90:
        return aqi_levels[3]
    return aqi_levels[4]


def fetch_and_predict(**kwargs):
    station = kwargs["station"]
    ts = datetime.fromisoformat(kwargs["ts"]).astimezone(pytz.timezone("Asia/Bangkok"))
    logging.info(f"ts: {ts}")
    df = fetch_history(station, ts)
    device = "cpu"
    pm_model = initialize_model(device, station)
    x, pred_time = process_input(df)
    y_pred = predict(pm_model, x, device)
    logging.info(f"Time: {pred_time} Prediction: {y_pred}")

    bucket_name = "ngae-datasci-project-2"
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob_name = f"pm2.5-predictions/{station}_hourly.csv"
    blob = bucket.blob(blob_name)
    content = blob.download_as_text()
    content += f"{pred_time},{pred_time.hour},{y_pred},{get_AQI(y_pred)}\n"
    logging.info(f"NEW DATA: {content}")
    blob.upload_from_string(content)
    blob.metadata = {"Cache-Control": "no-cache, no-store, max-age=0"}
    blob.patch()
    logging.info(f"Uploaded to {blob_name}")


default_args = {
    "depends_on_past": True,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    "predict_pm2_5",
    default_args=default_args,
    start_date=datetime(2022, 3, 1, 23, 0, 0, tzinfo=pytz.timezone("Asia/Bangkok")),
    schedule_interval="0 * * * *",
) as dag:
    start_task = EmptyOperator(task_id="start")
    end_task = EmptyOperator(task_id="end")

    stations = ["36t", "75t", "76t"]

    for station in stations:
        fetch_and_predict_task_id = f"fetch_and_predict_{station}"
        fetch_and_predict_task = PythonOperator(
            task_id=fetch_and_predict_task_id,
            python_callable=fetch_and_predict,
            op_kwargs={"station": station},
        )

        start_task >> fetch_and_predict_task >> end_task
