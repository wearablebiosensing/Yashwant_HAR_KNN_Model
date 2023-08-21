import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
import dash_core_components as dcc
import dash_html_components as html

 
# Read the CSV files and store them in a dictionary with file names as keys
file_names = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']
data_dict = {}
for file_name in file_names:
    df = pd.read_csv(f'{file_name}.csv')
    data_dict[file_name] = df

# Create a dictionary to map feature names to column indices
feature_columns = {
    'Min': 3,
    'Max': 4,
    'Std. Deviation': 5,
    'Average': 6,
    'Mean': 7,
    'Median': 8,
    'Kurtosis': 9,
    'Skewness': 10
}
activity_names = {
    1: 'WALKING',
    2: 'WALKING_UPSTAIRS',
    3: 'WALKING_DOWNSTAIRS',
    4: 'SITTING',
    5: 'STANDING',
    6: 'LAYING'
}

# Create a dropdown menu for file selection
file_dropdown = [{'label': file_name, 'value': file_name} for file_name in file_names]

# Create a dropdown menu for feature selection
feature_dropdown = [{'label': feature_name, 'value': feature_name} for feature_name in feature_columns.keys()]

# Create the app layout
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Histogram with Box Plots'),
    dcc.Dropdown(
        id='file-dropdown',
        options=file_dropdown,
        value=file_names[0],
        style={'width': '50%'}
    ),
    dcc.Dropdown(
        id='feature-dropdown',
        options=feature_dropdown,
        value='Std. Deviation',
        style={'width': '50%'}
    ),
    dcc.Graph(id='histogram-box-plot')
])

@app.callback(
    dash.dependencies.Output('histogram-box-plot', 'figure'),
    [dash.dependencies.Input('file-dropdown', 'value'),
     dash.dependencies.Input('feature-dropdown', 'value')]
)
def update_graph(selected_file, selected_feature):
    df = data_dict[selected_file]
    x_column = ' Activity ID'
    y_column = df.columns[feature_columns[selected_feature]]

    df[x_column] = df[x_column].map(activity_names)
    
    fig = make_subplots(rows=1, cols=1)

    fig.add_trace(go.Box(
        x=df[x_column],
        y=df[y_column],
        boxpoints='all',
        jitter=0.3,
        pointpos=-1.8,
        name=f'{selected_file} - {selected_feature}',
        marker=dict(
            color='rgb(7,40,89)',
        ),
        line=dict(
            color='rgb(7,40,89)',
        ),
        boxmean='sd'
    ))

    fig.update_layout(
        title=f'{selected_file} - {selected_feature} Histogram with Box Plot',
        xaxis_title=x_column,
        yaxis_title=selected_feature,
        showlegend=False
    )

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
