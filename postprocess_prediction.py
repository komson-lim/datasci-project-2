import pandas as pd

aqi_levels = ['Very Good', 'Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy']
stations = ['75t']

def get_level(x):
    if x < 25:
        return aqi_levels[0]
    elif x < 37:
        return aqi_levels[1]
    elif x < 50:
        return aqi_levels[2]
    elif x < 90:
        return aqi_levels[3]
    else: return aqi_levels[0]

def add_AQI_column(df):
    df['AQI'] = df['PM2_5'].apply(get_level)

for s in stations:
    df = pd.read_csv('prediction_result/' + s +'_prediction.csv')
    df['Datetime'] = pd.to_datetime(df['Datetime'])

    df_hourly = df.copy()
    df_hourly['Hour'] = df_hourly['Datetime'].dt.hour
    add_AQI_column(df_hourly)
    df_hourly = df_hourly[['Datetime', 'Hour', 'PM2_5', 'AQI']]
    df_hourly.to_csv('viz_source/'+ s +'_hourly.csv', index=False)

    df_weekly = df.resample('W', on='Datetime').mean()
    df_weekly.reset_index(inplace=True)
    df_weekly['Week Number'] = df_weekly['Datetime'].dt.isocalendar().week
    add_AQI_column(df_weekly)
    df_weekly = df_weekly[['Datetime', 'Week Number', 'PM2_5', 'AQI']]
    df_weekly.to_csv('viz_source/'+ s +'_weekly.csv', index=False)

    