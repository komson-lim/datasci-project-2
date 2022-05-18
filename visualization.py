import streamlit as st
import pandas as pd
import altair as alt

# reload on csv update
# https://discuss.streamlit.io/t/how-to-monitor-the-filesystem-and-have-streamlit-updated-when-some-files-are-modified/822/8

'''
# Thailand PM2.5 Forecasting 
'''

df = pd.read_csv('75t_prediction.csv', parse_dates=True)
aqi_levels = ['Very Good', 'Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy']
aqi_colors = ['#03d3fc', '#80e85a', '#ffdf29', '#ff9729', '#ff3e29']

@st.cache
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

df['AQI'] = df['PM2_5'].apply(get_level)

with st.sidebar:
    st.header('AQI Legend')
    for i in range(5):
        st.color_picker(label=aqi_levels[i], value=aqi_colors[i], disabled=True)

if st.checkbox('Show dataframe'):
    df
    
'''
## Overall Trend Chart

Select an interval on the top graph to zoom in! Refer to the sidebar for the AQI legend.
'''
selection = alt.selection(type='interval', encodings=['x'])

base = alt.Chart(df).mark_line(color='slategray').encode(
    x=alt.X('Datetime:T', title='Date', axis=alt.Axis(format='%b %y', grid=True)),
    y=alt.Y('PM2_5:Q', title='PM2.5 (µg/m3)')
).properties(
    width=800,
    height=300
)

# Display different color backgrounds by AQI, and show lines at cutoffs
def get_AQI_visuals():
    line0 = alt.Chart(df).mark_rule(color='gray', strokeDash=[12, 6], size=2).encode(y=alt.datum(25.0))
    line1 = alt.Chart(df).mark_rule(color='gray', strokeDash=[12, 6], size=2).encode(y=alt.datum(37.0))
    line2 = alt.Chart(df).mark_rule(color='gray', strokeDash=[12, 6], size=2).encode(y=alt.datum(50.0))
    line3 = alt.Chart(df).mark_rule(color='gray', strokeDash=[12, 6], size=2).encode(y=alt.datum(90.0))

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
can select station for every graph?

## PM2.5 by month
Line graph, show 3 years compared

## PM2.5 by time of day
Radial chart from 0-23
Filter by month, use average/max/min of that month's PM2.5

## PM2.5 by day of week
Radial chart from 0-23
Filter by month, use average/max/min of that month's PM2.5

'''
