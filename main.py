from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# Load and prepare data
df = pd.read_csv('TNM098_Lab2/EyeTrack-raw.tsv', sep='\t')
df = df.dropna(subset=['GazePointX(px)', 'GazePointY(px)', 'RecordingTimestamp'])
df['point_id'] = np.arange(len(df))
MAX_MARKER_SIZE = 30

app = Dash(__name__)

# Layout
app.layout = html.Div(
    style={'display': 'flex', 'height': '100vh', 'padding': '10px'},
    children=[
        html.Div(
            style={'width': '18%', 'paddingRight': '10px'},
            children=[
                html.P("Timestamp range (ms):", style={'fontSize': '12px'}),
                dcc.RangeSlider(
                    id='range-slider',
                    min=0, max=281000, step=1000,
                    marks={0: '0', 100000: '100k', 200000: '200k', 281000: '281k'},
                    value=[0, 281000]
                ),
                html.Br(),
                dcc.Checklist(
                    id='cluster-toggle',
                    options=[{'label': ' Show Clusters', 'value': 'enable'}],
                    value=[],
                    style={'fontSize': '12px'}
                ),
                dcc.Checklist(
                    id='show-unclustered',
                    options=[{'label': ' Show unclustered points', 'value': 'show'}],
                    value=[],  # Default to unchecked
                    style={'fontSize': '12px'}
                ),
                html.P("Number of clusters (≥700 ms):", style={'fontSize': '12px', 'marginTop': '10px'}),
                dcc.Slider(
                    id='cluster-count',
                    min=2, max=5, step=1, value=3,
                    marks={i: str(i) for i in range(2, 11)}
                ),
                dcc.Store(id='brush-ids', data=[]),
                dcc.Store(id='hover-ids', data=[])
            ]
        ),
        html.Div(
            style={'width': '82%'},
            children=[
                html.Div(
                    style={'display': 'flex', 'justifyContent': 'space-between'},
                    children=[
                        dcc.Graph(id='main-graph', style={'width': '49%', 'height': '45%'}),
                        dcc.Graph(id='cluster-graph', style={'width': '49%', 'height': '45%'})
                    ]
                ),
                dcc.Graph(id='timeline-graph', style={'height': '40%', 'marginTop': '10px'})
            ]
        )
    ]
)

# Brush callback
@app.callback(
    Output('brush-ids', 'data'),
    Input('main-graph', 'selectedData'),
    Input('cluster-graph', 'selectedData'),
    Input('timeline-graph', 'selectedData')
)
def update_brush(main_sel, cluster_sel, timeline_sel):
    ids = set()
    for sel in (main_sel, cluster_sel, timeline_sel):
        if sel and 'points' in sel:
            for pt in sel['points']:
                pid = pt['customdata'][0] if 'customdata' in pt else df.iloc[pt['pointIndex']]['point_id']
                ids.add(int(pid))
    return list(ids)

# Hover callback
@app.callback(
    Output('hover-ids', 'data'),
    Input('main-graph', 'hoverData'),
    Input('cluster-graph', 'hoverData'),
    Input('timeline-graph', 'hoverData')
)
def update_hover(main_hov, cluster_hov, timeline_hov):
    ids = set()
    for hov in (main_hov, cluster_hov, timeline_hov):
        if hov and 'points' in hov:
            for pt in hov['points']:
                pid = pt['customdata'][0] if 'customdata' in pt else df.iloc[pt['pointIndex']]['point_id']
                ids.add(int(pid))
    return list(ids)

# Graph update callback
@app.callback(
    Output('main-graph', 'figure'),
    Output('cluster-graph', 'figure'),
    Output('timeline-graph', 'figure'),
    Input('range-slider', 'value'),
    Input('cluster-toggle', 'value'),
    Input('cluster-count', 'value'),
    Input('brush-ids', 'data'),
    Input('hover-ids', 'data'),
    Input('show-unclustered', 'value')
)
def update_figure(selected_range, cluster_toggle, k, brush_ids, hover_ids, show_unclustered):
    low, high = selected_range
    filtered_df = df[(df['RecordingTimestamp'] >= low) & (df['RecordingTimestamp'] <= high)].copy()
    highlight_ids = sorted(set(brush_ids or []) | set(hover_ids or []))

    fig_main = px.scatter(
        filtered_df,
        x='GazePointX(px)', y='GazePointY(px)',
        size='GazeEventDuration(mS)',
        color='RecordingTimestamp',
        size_max=MAX_MARKER_SIZE,
        title="Eye Tracking Overview",
        color_continuous_scale='viridis',
        custom_data=['point_id']
    )
    fig_main.update_traces(
        selectedpoints=highlight_ids,
        unselected=dict(marker=dict(opacity=0.7)),
        selected=dict(marker=dict(opacity=1, size=MAX_MARKER_SIZE, color='red'))
    )

    fig_cluster = go.Figure().add_annotation(text="Clustering disabled", showarrow=False)
    fig_timeline = go.Figure().add_annotation(text="Clustering disabled", showarrow=False)

    if 'enable' in cluster_toggle and not filtered_df.empty:
        valid = filtered_df[filtered_df['GazeEventDuration(mS)'] >= 700].copy()
        if len(valid) >= k:
            kmeans = KMeans(n_clusters=k, random_state=0)
            valid['cluster'] = kmeans.fit_predict(valid[['GazePointX(px)', 'GazePointY(px)']])
            filtered_df = filtered_df.merge(valid[['point_id', 'cluster']], on='point_id', how='left')
        else:
            filtered_df['cluster'] = -1

        cluster_plot_df = filtered_df.copy()
        show_only_clustered = 'show' not in show_unclustered

        fig_cluster = px.scatter(
            cluster_plot_df.loc[cluster_plot_df['cluster'].notna()] if show_only_clustered else cluster_plot_df,
            x='GazePointX(px)', y='GazePointY(px)',
            size='GazeEventDuration(mS)', color='cluster',
            size_max=MAX_MARKER_SIZE,
            title="Cluster View",
            custom_data=['point_id']
        )

        fig_cluster.update_traces(
            selectedpoints=highlight_ids,
            unselected=dict(marker=dict(opacity=0.7)),
            selected=dict(marker=dict(opacity=1, size=MAX_MARKER_SIZE, color='red'))
        )

        fig_timeline = px.scatter(
            filtered_df,
            x='RecordingTimestamp', y='cluster',
            size='GazeEventDuration(mS)', color='cluster',
            size_max=MAX_MARKER_SIZE,
            title="Cluster Timeline",
            custom_data=['point_id']
        )
        fig_timeline.update_traces(
            selectedpoints=highlight_ids,
            unselected=dict(marker=dict(opacity=0.7)),
            selected=dict(marker=dict(opacity=1, size=MAX_MARKER_SIZE, color='red'))
        )

    return fig_main, fig_cluster, fig_timeline

=======
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
