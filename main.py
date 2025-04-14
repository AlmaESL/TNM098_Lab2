from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd

# Load data
df = pd.read_csv('TNM098_Lab2/EyeTrack-raw.tsv', sep='\t')

# Initialize app
app = Dash(__name__)

# App layout
app.layout = html.Div([
    html.H3('Eye Tracking Data Viewer'),
    
    dcc.Graph(id='graph'),

    html.P("Filter by timestamp range (ms):"),
    dcc.RangeSlider(
        id='range-slider',
        min=0,
        max=281000,
        value=[0, 281000],
        step=1000,
        marks={
            0: '0', 
            100000: '100k', 
            200000: '200k', 
            281000: '281k'
        },
        tooltip={"placement": "bottom", "always_visible": False}
    )
])

# Callback to update graph
@app.callback(
    Output('graph', 'figure'),
    Input('range-slider', 'value')
)
def update_figure(selected_range):
    low, high = selected_range
    mask = (df['RecordingTimestamp'] >= low) & (df['RecordingTimestamp'] <= high)
    filtered_df = df[mask]

    fig = px.scatter(filtered_df, 
                     x='GazePointX(px)', 
                     y='GazePointY(px)', 
                     size='GazeEventDuration(mS)', 
                     opacity=0.5, 
                     color="RecordingTimestamp",
                     color_continuous_scale=[[0.0, "#4F00A0"], [1.0, "#FFFF00"]],
                     range_color=[0, 281000])
    return fig

app.run(debug=True)