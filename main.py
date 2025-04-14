from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

# Load data
df = pd.read_csv('TNM098_Lab2/EyeTrack-raw.tsv', sep='\t')

# Initialize app
app = Dash(__name__)

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
        marks={0: '0', 100000: '100k', 200000: '200k', 281000: '281k'},
        tooltip={"placement": "bottom", "always_visible": False}
    ),

    html.Br(),

    html.P("Enable Clustering on FixationIndex:"),
    dcc.Checklist(
        id='cluster-toggle',
        options=[{'label': 'Show Clusters', 'value': 'enable'}],
        value=[]
    ),

    html.Div([
        html.P("Number of Clusters:"),
        dcc.Slider(id='cluster-count', min=1, max=5, step=1, value=3,
                   marks={i: str(i) for i in range(2, 11)})
    ], style={'marginTop': 10})
])

@app.callback(
    Output('graph', 'figure'),
    Input('range-slider', 'value'),
    Input('cluster-toggle', 'value'),
    Input('cluster-count', 'value')
)
def update_figure(selected_range, cluster_toggle, k):
    low, high = selected_range
    mask = (df['RecordingTimestamp'] >= low) & (df['RecordingTimestamp'] <= high)
    filtered_df = df[mask].copy()

    if 'enable' in cluster_toggle and not filtered_df.empty and 'FixationIndex' in filtered_df.columns:
        # Drop NaNs and reshape for clustering
        valid = filtered_df[['FixationIndex']].dropna()

        if len(valid) >= k:
            kmeans = KMeans(n_clusters=k, random_state=0)
            clusters = kmeans.fit_predict(valid)
            filtered_df.loc[valid.index, 'Cluster'] = clusters
            # Same color as the scatter plot on timestamp 
            color = 'Cluster'
        else:
            filtered_df['Cluster'] = np.nan
            color = 'RecordingTimestamp'
    else:
        color = 'RecordingTimestamp'

    fig = px.scatter(filtered_df,
                     x='GazePointX(px)',
                     y='GazePointY(px)',
                     size='GazeEventDuration(mS)',
                     opacity=0.6,
                     color=color,
                     color_continuous_scale='Viridis' if color == 'RecordingTimestamp' else None,
                     range_color=[0, 281000] if color == 'RecordingTimestamp' else None)

    return fig

# Run server
app.run(debug=True)
