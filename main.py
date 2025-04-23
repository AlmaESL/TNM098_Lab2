from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

# Load data
df = pd.read_csv('TNM098_Lab2/EyeTrack-raw.tsv', sep='\t')

# Initialize app
app = Dash(__name__)

# App layout
app.layout = html.Div(style={'display': 'flex', 'height': '100vh', 'padding': '10px'}, children=[

    # Sidebar controls
    html.Div(style={'width': '18%', 'paddingRight': '10px'}, children=[
        html.P("Timestamp range (ms):", style={'fontSize': '12px', 'marginBottom': '4px'}),
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
        html.P("Clustering options:", style={'fontSize': '12px', 'marginBottom': '4px'}),
        dcc.Checklist(
            id='cluster-toggle',
            options=[{'label': ' Show Clusters', 'value': 'enable'}],
            value=[],
            style={'fontSize': '12px'}
        ),
        html.P("Number of Clusters \n (GeoSpatial + Duration > 700ms):", style={'fontSize': '12px', 'marginTop': '10px'}),
        dcc.Slider(
            id='cluster-count',
            min=1,
            max=5,
            step=1,
            value=3,
            marks={i: str(i) for i in range(2, 11)},
            tooltip={"placement": "bottom", "always_visible": False}
        ),
    ]),

    # Graph area
    html.Div(style={'width': '82%'}, children=[
        html.Div([
            dcc.Graph(id='graph-main', style={'height': '45%', 'width': '49%'}),
            dcc.Graph(id='graph-cluster', style={'height': '45%', 'width': '49%'})
        ], style={'display': 'flex', 'justifyContent': 'space-between'}),
        dcc.Graph(id='graph-timeline', style={'height': '40%', 'marginTop': '10px'}),
    ])
])

# Callback function for updating the graphs
@app.callback(
    Output('graph-main', 'figure'),
    Output('graph-cluster', 'figure'),
    Output('graph-timeline', 'figure'),
    Input('range-slider', 'value'),
    Input('cluster-toggle', 'value'),
    Input('cluster-count', 'value')
)
def update_figure(selected_range, cluster_toggle, k):
    low, high = selected_range
    mask = (df['RecordingTimestamp'] >= low) & (df['RecordingTimestamp'] <= high)
    filtered_df = df[mask].copy()

    # Main scatter plot (eye tracking data with color based on timestamp)
    fig_main = px.scatter(
        filtered_df,
        x='GazePointX(px)',
        y='GazePointY(px)',
        size='GazeEventDuration(mS)',
        opacity=0.6,
        color='RecordingTimestamp',  # Color based on timestamp
        color_continuous_scale='Viridis',
        range_color=[0, 281000],
        title='Eye Tracking Data'
    )

    # Default empty cluster and timeline plots
    fig_cluster = px.scatter(title="Clustering not enabled.")
    fig_timeline = px.scatter(title="Clustering not enabled.")

    if 'enable' in cluster_toggle and not filtered_df.empty:
        # Filter valid gaze points with a minimum duration of 700ms
        valid = filtered_df[
            (filtered_df['GazeEventDuration(mS)'] >= 700)
        ][['RecordingTimestamp', 'GazePointX(px)', 'GazePointY(px)']].dropna()

        if len(valid) >= k:
            # Perform clustering only based on spatial coordinate proximity
            kmeans = KMeans(n_clusters=k, random_state=0)
            clusters = kmeans.fit_predict(valid[['GazePointX(px)', 'GazePointY(px)']])

            # Assign clusters back to the original dataframe
            filtered_df.loc[valid.index, 'Cluster'] = clusters

            # Mark unclustered points with NaN (they'll be assigned a low opacity)
            filtered_df['Cluster'].fillna(-1, inplace=True)

            # Set opacity for points: clustered points will have higher opacity, unclustered will be dimmed
            filtered_df['opacity'] = np.where(filtered_df['Cluster'] == -1, 0.1, 0.85)

            # Clustered points scatter plot
            fig_cluster = px.scatter(
                filtered_df,
                x='GazePointX(px)',
                y='GazePointY(px)',
                size='GazeEventDuration(mS)',
                opacity=filtered_df['opacity'],
                color='Cluster',  # Color based on cluster id
                color_discrete_map={i: px.colors.qualitative.Set1[i] for i in range(k)},  # Set distinct colors for clusters
                title=f'Cluster View'
            )

            # Cluster timeline plot
            fig_timeline = px.scatter(
                filtered_df,
                x='RecordingTimestamp',
                y='Cluster',
                color='Cluster',  # Color based on cluster id
                size='GazeEventDuration(mS)',
                color_discrete_map={i: px.colors.qualitative.Set1[i] for i in range(k)},  # Set distinct colors for clusters
                title='Cluster Timeline',
                labels={'RecordingTimestamp': 'Time (ms)', 'Cluster': 'Cluster ID'}
            )
        else:
            fig_cluster = px.scatter(title="Not enough data for clustering.")
            fig_timeline = px.scatter(title="Not enough data for clustering.")

    return fig_main, fig_cluster, fig_timeline

# Run app
app.run(debug=True)
