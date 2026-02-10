import dash
import dash.html as html
import dash.dcc as dcc
import dash_bootstrap_components as dbc
from dash import Output, Input, State, callback
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os
from pathlib import Path

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Determine the data directory path
current_file = Path(__file__).resolve()
project_root = current_file.parent
data_dir = project_root / 'data'

# Load data
poverty_data = pd.read_csv(data_dir / 'PovStatsData.csv')
poverty = pd.read_csv(data_dir / 'poverty.csv', low_memory=False)

# Define regions to exclude
regions = ['East Asia & Pacific', 'Europe & Central Asia',
           'Fragile and conflict affected situations', 'High income',
           'IDA countries classified as fragile situations', 'IDA total',
           'Latin America & Caribbean', 'Low & middle income', 'Low income',
           'Lower middle income', 'Middle East & North Africa',
           'Middle income', 'South Asia', 'Sub-Saharan Africa',
           'Upper middle income', 'World']

# Prepare data for logistic regression
def prepare_data():
    """Prepare data for logistic regression model"""
    # Get Gini data
    gini_data = poverty.dropna(subset=['GINI index (World Bank estimate)']).copy()
    
    # Extract relevant features
    features_to_use = ['GINI index (World Bank estimate)']
    
    # Get additional features if available
    income_cols = [col for col in gini_data.columns if 'Income share' in col]
    features_to_use.extend(income_cols[:3])  # Use first 3 income share columns
    
    # Prepare dataset
    df = gini_data[['Country Name', 'year'] + features_to_use].dropna()
    
    if df.empty or len(df) < 10:
        # Fallback: use Gini data only
        df = gini_data[['Country Name', 'year', 'GINI index (World Bank estimate)']].dropna()
        features = ['GINI index (World Bank estimate)']
    else:
        features = features_to_use
    
    # Create binary target: high poverty (GINI > median)
    gini_col = 'GINI index (World Bank estimate)'
    df['high_poverty'] = (df[gini_col] > df[gini_col].median()).astype(int)
    
    return df, features, gini_col

# Prepare the data
df_model, features, gini_col = prepare_data()

# Train logistic regression model
X = df_model[features].values
y = df_model['high_poverty'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

# Get predictions
y_pred = log_reg.predict(X_test_scaled)
y_pred_proba = log_reg.predict_proba(X_test_scaled)[:, 1]

# Calculate metrics
conf_matrix = confusion_matrix(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1('Logistic Regression: Poverty Rate Classification', 
                    className='text-center mb-4 mt-4'),
            html.H4('Predicting High Poverty Based on Economic Indicators', 
                   className='text-center text-muted mb-4'),
        ])
    ]),
    
    html.Hr(),
    
    # Model Information Tab
    dbc.Tabs([
        # Tab 1: Model Performance
        dbc.Tab([
            dbc.Row([
                dbc.Col([
                    html.H3('Model Performance Metrics', className='mt-4'),
                    dbc.Card([
                        dbc.CardBody([
                            html.H5('Accuracy on Test Set', className='card-title'),
                            html.H2(f"{log_reg.score(X_test_scaled, y_test):.3f}", 
                                   className='text-success'),
                        ])
                    ], className='mb-3'),
                    dbc.Card([
                        dbc.CardBody([
                            html.H5('ROC AUC Score', className='card-title'),
                            html.H2(f"{roc_auc:.3f}", className='text-info'),
                        ])
                    ], className='mb-3'),
                ], md=3),
                
                dbc.Col([
                    html.H3('Confusion Matrix', className='mt-4'),
                    dcc.Graph(
                        figure=go.Figure(data=go.Heatmap(
                            z=conf_matrix,
                            x=['Low Poverty', 'High Poverty'],
                            y=['Predicted Low', 'Predicted High'],
                            colorscale='Blues',
                            text=conf_matrix,
                            texttemplate='%{text}',
                            textfont={"size": 16},
                        )).update_layout(
                            title='Confusion Matrix',
                            xaxis_title='True Label',
                            yaxis_title='Predicted Label',
                            height=400
                        )
                    )
                ], md=4),
                
                dbc.Col([
                    html.H3('ROC Curve', className='mt-4'),
                    dcc.Graph(
                        figure=go.Figure([
                            go.Scatter(x=fpr, y=tpr, name=f'ROC (AUC={roc_auc:.3f})',
                                      line=dict(color='blue', width=2)),
                            go.Scatter(x=[0, 1], y=[0, 1], name='Random Classifier',
                                      line=dict(color='red', width=2, dash='dash'))
                        ]).update_layout(
                            title='ROC Curve',
                            xaxis_title='False Positive Rate',
                            yaxis_title='True Positive Rate',
                            hovermode='closest',
                            height=400
                        )
                    )
                ], md=5),
            ], className='mb-4'),
        ], label='Model Performance', tab_id='performance'),
        
        # Tab 2: Feature Importance
        dbc.Tab([
            dbc.Row([
                dbc.Col([
                    html.H3('Feature Coefficients', className='mt-4'),
                    dcc.Graph(
                        figure=px.bar(
                            x=log_reg.coef_[0],
                            y=features,
                            orientation='h',
                            labels={'x': 'Coefficient Value', 'y': 'Features'},
                            color=log_reg.coef_[0],
                            color_continuous_scale='RdBu',
                        ).update_layout(
                            title='Logistic Regression Coefficients',
                            height=400,
                            showlegend=False,
                        )
                    )
                ], md=12),
            ], className='mb-4'),
        ], label='Feature Importance'),
        
        # Tab 3: Make Predictions
        dbc.Tab([
            dbc.Row([
                dbc.Col([
                    html.H3('Predict High Poverty for New Data', className='mt-4'),
                    dbc.Card([
                        dbc.CardBody([
                            html.P('Enter values for the features below to predict poverty classification:'),
                            html.Br(),
                            
                            # Create input fields for each feature
                            html.Div(id='feature-inputs', children=[
                                dbc.Row([
                                    dbc.Col([
                                        html.Label(f'{feat}:', className='fw-bold'),
                                        dcc.Input(
                                            id={'type': 'feature-input', 'index': i},
                                            type='number',
                                            placeholder=f'Enter {feat}',
                                            className='form-control mb-3',
                                            style={'width': '100%'}
                                        )
                                    ], md=4)
                                    for i, feat in enumerate(features)
                                ]),
                            ], className='mb-3'),
                            
                            html.Br(),
                            dbc.Button('Predict', id='predict-button', color='primary', 
                                      className='mb-3', size='lg'),
                        ])
                    ]),
                ], md=6),
                
                dbc.Col([
                    html.H3('Prediction Result', className='mt-4'),
                    dbc.Card([
                        dbc.CardBody([
                            html.Div(id='prediction-output', children=[
                                html.P('Enter values and click Predict to see results.', 
                                      className='text-muted')
                            ])
                        ])
                    ], id='result-card'),
                ], md=6),
            ], className='mb-4'),
            
            dbc.Row([
                dbc.Col([
                    html.H4('Feature Ranges (from training data):', className='mt-4'),
                    html.Div([
                        html.P(f'{feat}: {X[:, i].min():.2f} - {X[:, i].max():.2f}')
                        for i, feat in enumerate(features)
                    ], className='text-small text-muted')
                ], md=12),
            ]),
        ], label='Make Predictions'),
        
        # Tab 4: Data Explorer
        dbc.Tab([
            dbc.Row([
                dbc.Col([
                    html.H3('Data Explorer', className='mt-4'),
                    html.P(f'Total samples: {len(df_model)}'),
                    html.P(f'High poverty: {(df_model["high_poverty"]==1).sum()}'),
                    html.P(f'Low poverty: {(df_model["high_poverty"]==0).sum()}'),
                ], md=3),
                
                dbc.Col([
                    html.H3('Data Distribution', className='mt-4'),
                    dcc.Graph(
                        figure=px.pie(
                            values=df_model['high_poverty'].value_counts().values,
                            names=['Low Poverty', 'High Poverty'],
                            color_discrete_sequence=['green', 'red']
                        ).update_layout(height=400)
                    )
                ], md=4),
                
                dbc.Col([
                    html.H3('Gini Index Distribution', className='mt-4'),
                    dcc.Graph(
                        figure=px.histogram(
                            df_model, x=gini_col,
                            nbins=30,
                            title='Distribution of Gini Index'
                        ).update_layout(height=400)
                    )
                ], md=5),
            ], className='mb-4'),
        ], label='Data Explorer'),
        
        # Tab 5: Project Info
        dbc.Tab([
            dbc.Row([
                dbc.Col([
                    html.H3('About This Project', className='mt-4'),
                    html.Ul([
                        html.Br(),
                        html.Li([
                            'Book: ',
                            html.B('Interactive Dashboards and Data Apps with Plotly and Dash')
                        ]),
                        html.Li([
                            'Publisher: ',
                            html.A('Packt Publishing',
                                  href='https://www.packtpub.com')
                        ]),
                        html.Li([
                            'GitHub Repository: ',
                            html.A('Interactive-Dashboards-and-Data-Apps-with-Plotly-and-Dash',
                                  href='https://github.com/PacktPublishing/Interactive-Dashboards-and-Data-Apps-with-Plotly-and-Dash')
                        ]),
                        html.Br(),
                        html.Li('Dataset: World Bank Poverty and Equity Database'),
                        html.Li([
                            'Data Source: ',
                            html.A('https://datacatalog.worldbank.org/dataset/poverty-and-equity-database',
                                  href='https://datacatalog.worldbank.org/dataset/poverty-and-equity-database')
                        ]),
                        html.Br(),
                        html.H5('Model Information:'),
                        html.Li('Algorithm: Logistic Regression'),
                        html.Li(f'Test Set Size: {len(X_test)} samples'),
                        html.Li(f'Number of Features: {len(features)}'),
                        html.Li('Scaler: StandardScaler'),
                    ])
                ], md=8),
            ], className='mb-4'),
        ], label='Project Info'),
    ]),
    
], fluid=True, className='mb-5')

# Callback for prediction
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    [State({'type': 'feature-input', 'index': dcc.ALL}, 'value')],
    prevent_initial_call=True
)
def make_prediction(n_clicks, input_values):
    """Make prediction based on user input"""
    if n_clicks is None or not input_values or any(val is None for val in input_values):
        raise PreventUpdate
    
    try:
        # Prepare input
        input_array = np.array([input_values]).reshape(1, -1)
        
        # Scale input
        input_scaled = scaler.transform(input_array)
        
        # Make prediction
        prediction = log_reg.predict(input_scaled)[0]
        probability = log_reg.predict_proba(input_scaled)[0]
        
        # Return result
        prediction_text = 'HIGH POVERTY (Gini > Median)' if prediction == 1 else 'LOW POVERTY (Gini ≤ Median)'
        prediction_color = 'danger' if prediction == 1 else 'success'
        confidence = probability[prediction]
        
        return [
            dbc.Card([
                dbc.CardBody([
                    html.H4('Prediction:', className='card-title'),
                    html.H3(prediction_text, className=f'text-{prediction_color}'),
                    html.Hr(),
                    html.P(f'Confidence: {confidence:.1%}', className='fw-bold'),
                    html.P(f'Probability of High Poverty: {probability[1]:.1%}'),
                    html.P(f'Probability of Low Poverty: {probability[0]:.1%}'),
                ])
            ])
        ]
    except Exception as e:
        return html.P(f'Error: {str(e)}', className='text-danger')


if __name__ == '__main__':
    app.run_server(debug=True)
