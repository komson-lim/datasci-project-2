from unicodedata import name
from jinja2 import Undefined
import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime

# reload on csv update
# https://docs.streamlit.io/library/api-reference/control-flow/st.experimental_rerun

'''
# Thailand PM2.5 Forecasting 
'''
# Constants and cached functions are here

aqi_levels = ['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy']
aqi_colors = ['#80e85a', '#ffdf29', '#ff9729', '#ff3e29']
aqi_cutoffs = [0.0, 12.0, 35.4, 55.4, 100]
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

@st.cache(ttl=1800)
def date_to_datetime(date):
    return datetime(date.year, date.month, date.day, 1,0)

@st.cache(ttl=1800)
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

# Begin app

with st.sidebar:
    st.header('Locations')
    for key in stations.keys():
        if st.checkbox(stations[key], value=True):
            visible_stations.append(key)

    st.header('AQI Legend')
    for i in range(4):
        st.color_picker(label=aqi_levels[i], value=aqi_colors[i], disabled=True)

df_hourly = load_data('75t', 'hourly')
df_weekly = load_data('75t', 'weekly')

message = 'Selected locations:'
for item in visible_stations:
    message += ' ' + stations[item]
message

st.write('## Latest prediction:')
latest_data = get_latest_data_item(df_hourly)
latest_data

'''
## PM2.5 all year round
'''
# add graph title

base1 = alt.Chart(df_hourly.iloc[-24:]).encode(
    x=alt.X('Datetime:T', title='Date', axis=alt.Axis(format='%e %b %y %H:%M', grid=True)),
    y=alt.Y('PM2_5:Q', title='Mean PM2.5 (µg/m3)'),
    color=alt.Color('AQI:O', scale=alt.Scale(domain=aqi_levels, range=aqi_colors))
)

history24h_chart = base1.mark_bar(width=15)
history24h_labels = base1.mark_text(baseline='top').encode(text=alt.Text('PM2_5:O', format=',.1f'))
st.altair_chart(history24h_chart + history24h_labels, use_container_width=True)


'''
## Overall Trend Chart
Select an interval on the top graph to zoom in! 
The colors display the Air Quality Index (AQI) level. Refer to the sidebar for the AQI legend.
'''
selection = alt.selection(type='interval', encodings=['x'])

base2 = alt.Chart(df_hourly).mark_line(color='slategray').encode(
    x=alt.X('Datetime:T', title='Date', axis=alt.Axis(format='%b %y', grid=True)),
    y=alt.Y('PM2_5:Q', title='PM2.5 (µg/m3)')
).properties(
    width=800,
    height=300
)

# Display different color backgrounds by AQI, and show lines at cutoffs
def get_AQI_visuals():
    visual = alt.Chart(pd.DataFrame({'y': [aqi_cutoffs[0]], 'y2':[aqi_cutoffs[1]]})).mark_rect(
        color=aqi_colors[0], opacity=0.2).encode(y='y', y2='y2'
    )
    
    for i in range(1,4):
        visual += alt.Chart(df_hourly).mark_rule(color='gray', strokeDash=[12, 6], size=2).encode(y=alt.datum(aqi_cutoffs[i]))
        visual += alt.Chart(pd.DataFrame({'y': [aqi_cutoffs[i]], 'y2':[aqi_cutoffs[i+1]]})).mark_rect(
            color=aqi_colors[i], opacity=0.2).encode(y='y', y2='y2'
        )

    return visual

overall_trend_chart_zoomed = base2.encode(
    x=alt.X('Datetime:T', scale=alt.Scale(domain=selection), title='Date', axis=alt.Axis(format='%e %b %y'))
)
overall_trend_chart = base2.properties(
    height=50
).add_selection(selection)
st.altair_chart(overall_trend_chart & (get_AQI_visuals() + overall_trend_chart_zoomed), use_container_width=True)

'''
## PM2.5 by time of day
Maximum, average, and minimum PM2.5 at each hour of day, calculated during the selected range.

Select a date range:
'''
start_date = st.date_input('Start date', 
    min_value=datetime(2020,6,2,1,0), 
    max_value=timestamp_to_datetime(latest_data['Datetime'][0]),
    value=datetime(2021,3,30,1,0))
end_date = st.date_input('End date', 
    min_value=datetime(2020,6,2,1,0), 
    max_value=timestamp_to_datetime(latest_data['Datetime'][0]),
    value=datetime(2021,3,31,1,0))

df_sliced = df_hourly[
    (pd.to_datetime(df_hourly['Datetime']) >= date_to_datetime(start_date)) 
    & (pd.to_datetime(df_hourly['Datetime']) <= date_to_datetime(end_date))]

df_sliced_mean = df_sliced.groupby('Hour').mean().reset_index()
add_AQI_column(df_sliced_mean)

df_sliced_min = df_sliced.groupby('Hour').min().reset_index()
add_AQI_column(df_sliced_min)

df_sliced_max = df_sliced.groupby('Hour').max().reset_index()
add_AQI_column(df_sliced_max)

day_round_chart_max = alt.Chart(df_sliced_max).mark_arc(stroke='white').encode(
    theta=alt.Theta('Hour:O'),
    radius=alt.Radius('PM2_5:Q', title='Mean PM2.5 (µg/m3)'),
    color=alt.Color('AQI:O', scale=alt.Scale(domain=aqi_levels, range=aqi_colors))
).properties(
    title='Maximum PM2.5 at each hour of day'
)

day_round_chart_mean = alt.Chart(df_sliced_mean).mark_arc(stroke='white').encode(
    theta=alt.Theta('Hour:O'),
    radius=alt.Radius('PM2_5:Q', title='Mean PM2.5 (µg/m3)'),
    color=alt.Color('AQI:O', scale=alt.Scale(domain=aqi_levels, range=aqi_colors))
).properties(
    title='Average PM2.5 at each hour of day'
)

day_round_chart_min = alt.Chart(df_sliced_min).mark_arc(stroke='white').encode(
    theta=alt.Theta('Hour:O'),
    radius=alt.Radius('PM2_5:Q', title='Mean PM2.5 (µg/m3)'),
    color=alt.Color('AQI:O', scale=alt.Scale(domain=aqi_levels, range=aqi_colors))
).properties(
    title='Minimum PM2.5 at each hour of day'
)

def get_chart_labels(chart):
    return chart.mark_text(radiusOffset=20).encode(text=alt.Text('PM2_5:O', format=',.1f'))

'View graphs'

if st.checkbox('Maximum PM2.5 during date range'):
    st.altair_chart(day_round_chart_max + get_chart_labels(day_round_chart_max), use_container_width=True)
if st.checkbox('Average PM2.5 during date range', value=True):
    st.altair_chart(day_round_chart_mean + get_chart_labels(day_round_chart_mean), use_container_width=True)
if st.checkbox('Minimum PM2.5 during date range'):
    st.altair_chart(day_round_chart_min + get_chart_labels(day_round_chart_min), use_container_width=True)


