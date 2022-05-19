import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime

# reload on csv update
# https://discuss.streamlit.io/t/how-to-monitor-the-filesystem-and-have-streamlit-updated-when-some-files-are-modified/822/

'''
# Thailand PM2.5 Forecasting 
TODO : Can select at least 1 station

'''
# Constants and cached functions are here

aqi_levels = ['Very Good', 'Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy']
aqi_colors = ['#03d3fc', '#80e85a', '#ffdf29', '#ff9729', '#ff3e29']
stations = {
    '36t': 'เชียงใหม่',
    '75t': 'น่าน',
    '76t': 'ตาก'
}
visible_stations = []


@st.cache(ttl=1800)
def load_data(station, item_freq='hourly'):
    return pd.read_csv('viz_source/' + station + '_' + item_freq + '.csv', parse_dates=True)

@st.cache(ttl=1800)
def get_latest_data_item(df_hourly):
    return df_hourly.iloc[[-1]].reset_index().drop('index', axis=1)

@st.cache(ttl=1800)
def timestamp_to_datetime(timestamp):
    return pd.to_datetime(timestamp).to_pydatetime()

# Begin app

with st.sidebar:
    st.header('Locations')
    for key in stations.keys():
        if st.checkbox(stations[key]):
            visible_stations.append(key)

    st.header('AQI Legend')
    for i in range(5):
        st.color_picker(label=aqi_levels[i], value=aqi_colors[i], disabled=True)

df_hourly = load_data('75t', 'hourly')
df_weekly = load_data('75t', 'weekly')

st.write('Latest prediction:')
latest_data = get_latest_data_item(df_hourly)
latest_data

'''
## Overall Trend Chart
Select an interval on the top graph to zoom in! 
The colors display the Air Quality Index (AQI) level. Refer to the sidebar for the AQI legend.
'''
selection = alt.selection(type='interval', encodings=['x'])

base = alt.Chart(df_hourly).mark_line(color='slategray').encode(
    x=alt.X('Datetime:T', title='Date', axis=alt.Axis(format='%b %y', grid=True)),
    y=alt.Y('PM2_5:Q', title='PM2.5 (µg/m3)')
).properties(
    width=800,
    height=300
)

# Display different color backgrounds by AQI, and show lines at cutoffs
def get_AQI_visuals():
    line0 = alt.Chart(df_hourly).mark_rule(color='gray', strokeDash=[12, 6], size=2).encode(y=alt.datum(25.0))
    line1 = alt.Chart(df_hourly).mark_rule(color='gray', strokeDash=[12, 6], size=2).encode(y=alt.datum(37.0))
    line2 = alt.Chart(df_hourly).mark_rule(color='gray', strokeDash=[12, 6], size=2).encode(y=alt.datum(50.0))
    line3 = alt.Chart(df_hourly).mark_rule(color='gray', strokeDash=[12, 6], size=2).encode(y=alt.datum(90.0))

    area0 = alt.Chart(pd.DataFrame({'y': [0], 'y2':[25]})).mark_rect(color=aqi_colors[0], opacity=0.2).encode(y='y', y2='y2')
    area1 = alt.Chart(pd.DataFrame({'y': [25], 'y2':[37]})).mark_rect(color=aqi_colors[1], opacity=0.2).encode(y='y', y2='y2')
    area2 = alt.Chart(pd.DataFrame({'y': [37], 'y2':[50]})).mark_rect(color=aqi_colors[2], opacity=0.2).encode(y='y', y2='y2')
    area3 = alt.Chart(pd.DataFrame({'y': [50], 'y2':[90]})).mark_rect(color=aqi_colors[3], opacity=0.2).encode(y='y', y2='y2')
    area4 = alt.Chart(pd.DataFrame({'y': [90], 'y2':[100]})).mark_rect(color=aqi_colors[4], opacity=0.2).encode(y='y', y2='y2')

    return line0 + line1 + line2 + line3 + area0 + area1 + area2 + area3 + area4

overall_trend_chart_zoomed = base.encode(
    x=alt.X('Datetime:T', scale=alt.Scale(domain=selection), title='Date', axis=alt.Axis(format='%e %b %y'))
)
overall_trend_chart = base.properties(
    height=50
).add_selection(selection)
st.altair_chart(overall_trend_chart & (get_AQI_visuals() + overall_trend_chart_zoomed), use_container_width=True)


'''
## PM2.5 by week
Radial chart/histogram 
'''


'''
## PM2.5 by time of day
Radial chart from 0-23
Filter by datetime range, use average/max/min of that month's PM2.5

'''
date_range = st.slider('Select a date range', 
    min_value=datetime(2020,6,2,1,0), 
    max_value=timestamp_to_datetime(latest_data['Datetime'][0]),
    value=(datetime(2020,6,2,1,0), datetime(2021,6,2,1,0)))
