import pandas as pd

aqi_levels = ['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy']
aqi_cutoffs = [0.0, 12.0, 35.4, 55.4, 100]
stations = ['75t']

def get_level(x):
    if x < aqi_cutoffs[1]:
        return aqi_levels[0]

    elif x < aqi_cutoffs[2]:
        return aqi_levels[1]

    elif x < aqi_cutoffs[3]:
        return aqi_levels[2]

    else: return aqi_levels[3]

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

    # df_weekly = df.resample('W', on='Datetime').mean()
    # df_weekly.reset_index(inplace=True)
    # df_weekly['Week Number'] = df_weekly['Datetime'].dt.isocalendar().week
    # df_weekly = df_weekly.groupby('Week Number').mean().reset_index()
    # add_AQI_column(df_weekly)
    # df_weekly.to_csv('viz_source/'+ s +'_weekly.csv', index=False)

    