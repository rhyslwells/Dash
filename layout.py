"""Dashboard layout definition"""

import dash.html as html
import dash.dcc as dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px


def create_layout(model, df_model, gini_col, features):
    """Create the dashboard layout"""
    
    conf_matrix = model.get_confusion_matrix()
    fpr, tpr, roc_auc = model.get_roc_curve()
    accuracy = model.get_accuracy()
    coefficients = model.get_coefficients()
    X = df_model[features].values
    
    layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1('Logistic Regression: Poverty Rate Classification', 
                        className='text-center mb-4 mt-4'),
                html.H4('Predicting High Poverty Based on Economic Indicators', 
                       className='text-center text-muted mb-4'),
            ])
        ]),
        
        html.Hr(),
        
        # Model Information Tabs
        dbc.Tabs([
            # Tab 1: Model Performance
            dbc.Tab([
                dbc.Row([
                    dbc.Col([
                        html.H3('Model Performance Metrics', className='mt-4'),
                        dbc.Card([
                            dbc.CardBody([
                                html.H5('Accuracy on Test Set', className='card-title'),
                                html.H2(f"{accuracy:.3f}", 
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
                                x=coefficients,
                                y=features,
                                orientation='h',
                                labels={'x': 'Coefficient Value', 'y': 'Features'},
                                color=coefficients,
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
                            html.Li(f'Test Set Size: {len(model.X_test)} samples'),
                            html.Li(f'Number of Features: {len(features)}'),
                            html.Li('Scaler: StandardScaler'),
                        ])
                    ], md=8),
                ], className='mb-4'),
            ], label='Project Info'),
        ]),
        
    ], fluid=True, className='mb-5')
    
    return layout
