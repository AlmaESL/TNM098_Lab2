import dash
from dash import Dash, dcc, html, Input, Output, State
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------------------------------------------
# 1. Load & prep
# -------------------------------------------------------------------
df = pd.read_csv('TNM098_Lab2/EyeTrack-raw.tsv', sep='\t')
df = df.dropna(subset=['GazePointX(px)', 'GazePointY(px)', 'RecordingTimestamp'])
df['point_id'] = np.arange(len(df))

# Maximum marker size for scatter plots
MAX_MARKER_SIZE = 30

# -------------------------------------------------------------------
# 2. App & layout
# -------------------------------------------------------------------
app = Dash(__name__)

app.layout = html.Div(
    style={'display': 'flex', 'height': '100vh', 'padding': '10px'},
    children=[
        # Sidebar controls
        html.Div(
            style={'width': '18%', 'paddingRight': '10px'},
            children=[
                html.P("Timestamp range (ms):", style={'fontSize': '12px'}),
                dcc.RangeSlider(
                    id='range-slider', min=0, max=281000, step=1000,
                    marks={0: '0', 100000: '100k', 200000: '200k', 281000: '281k'},
                    value=[0, 281000]
                ),
                html.Br(),
                dcc.Checklist(
                    id='cluster-toggle',
                    options=[{'label': ' Show Clusters', 'value': 'enable'}],
                    value=[]
                ),
                html.P("Number of clusters (≥700 ms):", style={'fontSize': '12px', 'marginTop': '10px'}),
                dcc.Slider(
                    id='cluster-count', min=1, max=5, step=1, value=3,
                    marks={i: str(i) for i in range(1, 6)}
                ),
            ]
        ),

        # Stores for brush vs hover
        dcc.Store(id='brush-ids', data=[]),
        dcc.Store(id='hover-ids', data=[]),

        # Graph area
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

# -------------------------------------------------------------------
# 3. Callbacks for brush & hover
# -------------------------------------------------------------------
@app.callback(
    Output('brush-ids', 'data'),
    [
        Input('main-graph', 'selectedData'),
        Input('cluster-graph', 'selectedData'),
        Input('timeline-graph', 'selectedData'),
    ]
)
def update_brush(main_sel, cluster_sel, timeline_sel):
    ids = set()
    for ev in (main_sel, cluster_sel, timeline_sel):
        if ev and 'points' in ev:
            for pt in ev['points']:
                pid = df.iloc[pt['pointIndex']]['point_id']
                ids.add(int(pid))
    return list(ids)

@app.callback(
    Output('hover-ids', 'data'),
    [
        Input('main-graph', 'hoverData'),
        Input('cluster-graph', 'hoverData'),
        Input('timeline-graph', 'hoverData'),
    ]
)
def update_hover(main_hov, cluster_hov, timeline_hov):
    ids = set()
    for ev in (main_hov, cluster_hov, timeline_hov):
        if ev and 'points' in ev:
            for pt in ev['points']:
                pid = df.iloc[pt['pointIndex']]['point_id']
                ids.add(int(pid))
    return list(ids)

# -------------------------------------------------------------------
# 4. Helper for scatter with capped point sizes and highlighting
# -------------------------------------------------------------------
def make_scatter(df, x, y, size=None, color=None, title="", highlight_ids=None, color_continuous_scale=None):
    fig = px.scatter(
        df,
        x=x,
        y=y,
        size=size,
        color=color,
        size_max=MAX_MARKER_SIZE,
        title=title,
        color_continuous_scale=color_continuous_scale
    )
    if highlight_ids is not None:
        fig.update_traces(
            selectedpoints=highlight_ids,
            unselected=dict(marker=dict(opacity=0.7)),
            selected=dict(marker=dict(opacity=1, size=MAX_MARKER_SIZE, color='red'))
        )
    return fig

# -------------------------------------------------------------------
# 5. Main redraw callback
# -------------------------------------------------------------------
@app.callback(
    Output('main-graph', 'figure'),
    Output('cluster-graph', 'figure'),
    Output('timeline-graph', 'figure'),
    [
        Input('range-slider', 'value'),
        Input('cluster-toggle', 'value'),
        Input('cluster-count', 'value'),
        Input('brush-ids', 'data'),
        Input('hover-ids', 'data')
    ]
)
def update_all(ts_range, cluster_toggle, k, brush_ids, hover_ids):
    low, high = ts_range
    dff = df[(df['RecordingTimestamp'] >= low) & (df['RecordingTimestamp'] <= high)]

    # determine which points to highlight
    highlight = sorted(set(brush_ids or []) | set(hover_ids or []))

    # Main graph (time → color)
    fig_main = make_scatter(
        dff,
        x='GazePointX(px)',
        y='GazePointY(px)',
        size='GazeEventDuration(mS)',
        color='RecordingTimestamp',
        title="Eye Tracking Overview",
        highlight_ids=highlight,
        color_continuous_scale='viridis'
    )

    # default placeholders
    fig_cluster = go.Figure().add_annotation(text="Clustering disabled", showarrow=False)
    fig_timeline = go.Figure().add_annotation(text="Clustering disabled", showarrow=False)

    # If clustering is enabled
    if 'enable' in cluster_toggle:
        valid = dff[dff['GazeEventDuration(mS)'] >= 700].copy()
        if len(valid) >= k:
            valid['cluster'] = KMeans(n_clusters=k, random_state=0).fit_predict(
                valid[['GazePointX(px)', 'GazePointY(px)']]
            )
            dff = dff.merge(valid[['point_id', 'cluster']], on='point_id', how='left')
        else:
            dff['cluster'] = -1

        # Cluster view: build, then tweak each trace individually
        fig_cluster = px.scatter(
            dff,
            x='GazePointX(px)',
            y='GazePointY(px)',
            size='GazeEventDuration(mS)',
            color='cluster',
            size_max=MAX_MARKER_SIZE,
            title="Cluster View"
        )
        for trace in fig_cluster.data:
            # if cluster is nan - lower opacity
            if trace.name == 'NaN':
                trace.update(
                    marker=dict(opacity=0.1),
                    selectedpoints=highlight,
                    unselected=dict(marker=dict(opacity=0.1)),
                    selected=dict(marker=dict(opacity=1, size=MAX_MARKER_SIZE, color='red'))
                )
            else:
                # clustered = normal base, dim when unselected
                trace.update(
                    marker=dict(opacity=0.7),
                    selectedpoints=highlight,
                    unselected=dict(marker=dict(opacity=0.7)),
                    selected=dict(marker=dict(opacity=1, size=MAX_MARKER_SIZE, color='red'))
                )

        # Timeline view: same capped size & styling
        fig_timeline = px.scatter(
            dff,
            x='RecordingTimestamp',
            y='cluster',
            size='GazeEventDuration(mS)',
            color='cluster',
            size_max=MAX_MARKER_SIZE,
            title="Cluster Timeline"
        )
        fig_timeline.update_traces(
            selectedpoints=highlight,
            unselected=dict(marker=dict(opacity=0.7)),
            selected=dict(marker=dict(opacity=1, size=MAX_MARKER_SIZE, color='red'))
        )

    return fig_main, fig_cluster, fig_timeline

# -------------------------------------------------------------------
# 6. Run server
# -------------------------------------------------------------------
app.run(debug=True)
