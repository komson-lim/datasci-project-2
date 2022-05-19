import streamlit as st
import pandas as pd
import altair as alt

# reload on csv update
# https://docs.streamlit.io/library/api-reference/control-flow/st.experimental_rerun

'''
# Thailand PM2.5 Report
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
def load_data(station):
    return pd.read_csv('viz_source/' + station + '_hourly.csv', parse_dates=True)

@st.cache(ttl=1800)
def get_data_item(df_hourly, iloc):
    return df_hourly.iloc[[iloc]].reset_index().drop('index', axis=1)

@st.cache(ttl=1800)
def timestamp_to_date(timestamp):
    return pd.to_datetime(timestamp).date()

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

# TODO : Load all 3 files for 3 locations into a dict of df
# TODO : Writeup
# df_hourly = dict()
df_hourly = load_data('75t')

# remove
message = 'Selected locations:'
for item in visible_stations:
    message += ' ' + stations[item]
message

latest_date = get_data_item(df_hourly, -1)['Datetime'][0]
st.write('#### Latest prediction: ' + latest_date)

for s in visible_stations:
    first_data = get_data_item(df_hourly, 0)
    latest_data = get_data_item(df_hourly, -1)
    latest_data2 = get_data_item(df_hourly, -2)

    # for s in visible_stations:
    col1, col2, col3 = st.columns(3)
    col1.metric(label='Location', value=stations['75t'])
    col2.metric(
        label='PM2.5', 
        value=(str(round(latest_data['PM2_5'][0], 2)) + ' µg/m3'), 
        delta=round(latest_data['PM2_5'][0] - latest_data2['PM2_5'][0], 2),
        delta_color="inverse"
    )
    col3.metric(
        label='US AQI',
        value=latest_data['AQI'][0]
    )

'''
## PM2.5 in the Last 24 Hours
'''
for s in visible_stations:
    base = alt.Chart(df_hourly.iloc[-24:]).encode(
        x=alt.X('Datetime:T', title='Date', axis=alt.Axis(format='%e %b %Y %H:%M', grid=True)),
        y=alt.Y('PM2_5:Q', title='Mean PM2.5 (µg/m3)'),
    ).properties(
        title=stations[s]
    )

    history24h_chart = base.mark_bar(width=15).encode(
        color=alt.Color('AQI:O', scale=alt.Scale(domain=aqi_levels, range=aqi_colors))
    )
    history24h_labels = base.mark_text(align='center', baseline='bottom', dy=-10, color='slategray').encode(
        text=alt.Text('PM2_5:O', format=',.1f')
    )
    st.altair_chart(history24h_chart + history24h_labels, use_container_width=True)

'''
## PM2.5 by time of day

'''
date = st.date_input('Select a day:', 
    min_value=timestamp_to_date(first_data['Datetime'][0]), 
    max_value=timestamp_to_date(latest_data['Datetime'][0]),
    value=timestamp_to_date(latest_data['Datetime'][0]))

for s in visible_stations:
    df_sliced = df_hourly[(pd.to_datetime(df_hourly['Datetime']).dt.date == date)]

    base = alt.Chart(df_sliced).encode(
        theta=alt.Theta('Hour:O'),
        radius=alt.Radius('PM2_5:Q', title='Mean PM2.5 (µg/m3)'),
    ).properties(
        title=stations[s]
    )

    day_round_chart = base.mark_arc(stroke='white').encode(
        color=alt.Color('AQI:O', scale=alt.Scale(domain=aqi_levels, range=aqi_colors))
    )

    day_round_chart_labels = day_round_chart.mark_text(radiusOffset=15).encode(text=alt.Text('PM2_5:O', format=',.1f'))
    day_round_chart_hours = base.mark_text(radiusOffset=35, color='slategray').encode(text=alt.Text('Hour:O'))

    st.altair_chart(day_round_chart + day_round_chart_labels + day_round_chart_hours, use_container_width=True)

'''
## Overall Trend Chart
Select an interval on the top graph to zoom in! 
The colors display the Air Quality Index (AQI) level. Refer to the sidebar for the AQI legend.
'''
df_location = dict()
for s in visible_stations:
    df_location[s] = df_hourly.copy()
    df_location[s]['Location'] = stations[s]
df_merged = pd.concat([df_location[s] for s in visible_stations]) 

selection = alt.selection(type='interval', encodings=['x'])        

base = alt.Chart(df_merged).mark_line(color='slategray').encode(
    x=alt.X('Datetime:T', title='Date', axis=alt.Axis(format='%b %y', grid=True)),
    y=alt.Y('PM2_5:Q', title='PM2.5 (µg/m3)'),
    color=alt.Color('Location:N', scale=alt.Scale(domain=[stations[s] for s in visible_stations], range=['slategray','salmon','mediumaquamarine']))
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

overall_trend_chart_zoomed = base.encode(
    x=alt.X('Datetime:T', scale=alt.Scale(domain=selection), title='Date', axis=alt.Axis(format='%e %b %y'))
)
overall_trend_chart = base.properties(
    height=50
).add_selection(selection)
st.altair_chart(overall_trend_chart & (get_AQI_visuals() + overall_trend_chart_zoomed), use_container_width=True)
