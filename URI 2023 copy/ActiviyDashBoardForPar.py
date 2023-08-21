import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np



X_train = np.loadtxt('X_train.txt')
Y_train = np.loadtxt('y_train.txt')
X_test = np.loadtxt('X_test.txt')
Y_test = np.loadtxt('y_test.txt')
subject_train = np.loadtxt('subject_train.txt')


X_train_df = pd.DataFrame(X_train)
Y_train_df = pd.DataFrame(Y_train, columns=['Activity'])
subject_train_df = pd.DataFrame(subject_train, columns=['ParticipantID'])

import dash
import dash_core_components as dcc
import dash_html_components as html

import plotly.graph_objects as go
import pandas as pd
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
labels = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
]
# Prepare the data
X_train_df = pd.DataFrame(X_train)
Y_train_df = pd.DataFrame(Y_train, columns=['Activity'])
subject_train_df = pd.DataFrame(subject_train, columns=['ParticipantID'])

# Get unique participant IDs
participant_ids = subject_train_df['ParticipantID'].unique()

# Create the Dash app
app = dash.Dash(__name__)

# Define the app layout
app.layout = html.Div([
    dcc.Dropdown(
        id='participant-dropdown',
        options=[{'label': f'Participant {pid}', 'value': pid} for pid in participant_ids],
        value=participant_ids[0]
    ),
    html.Div(id='graphs-container', style={'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'center'})
])


@app.callback(
    dash.dependencies.Output('graphs-container', 'children'),
    [dash.dependencies.Input('participant-dropdown', 'value')]
)
def update_graphs(participant_id):
    # Filter the data based on the selected participant ID
    filtered_X = X_train_df.loc[subject_train_df['ParticipantID'] == participant_id]
    filtered_Y = Y_train_df.loc[subject_train_df['ParticipantID'] == participant_id]

    # Create the graphs
    graphs = []
    for activity_id in range(1, 7):
        activity_readings = filtered_X[filtered_Y['Activity'] == activity_id].iloc[:, :3]

        if len(activity_readings) > 0:
            fig = go.Figure()
            for col in activity_readings.columns:
                fig.add_trace(go.Scatter(y=activity_readings[col], mode='lines', name=col))
            fig.update_layout(
                title=f"Participant {participant_id} - Activity: {labels[activity_id - 1]}",
                xaxis_title='data points',
                yaxis_title='Acceleration',
                legend_title='Axis'
            )
            graph = dcc.Graph(figure=fig)
            graphs.append(graph)

    return graphs



if __name__ == '__main__':
    app.run_server(debug=True)
