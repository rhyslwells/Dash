"""Callbacks for dashboard interactivity"""

import numpy as np
import dash.html as html
import dash_bootstrap_components as dbc
from dash import Output, Input, State, callback, ALL
from dash.exceptions import PreventUpdate


def register_callbacks(app, model):
    """Register all callbacks for the dashboard"""
    
    @app.callback(
        Output('prediction-output', 'children'),
        Input('predict-button', 'n_clicks'),
        [State({'type': 'feature-input', 'index': ALL}, 'value')],
        prevent_initial_call=True
    )
    def make_prediction(n_clicks, input_values):
        """Make prediction based on user input"""
        if n_clicks is None or not input_values or any(val is None for val in input_values):
            raise PreventUpdate
        
        try:
            # Prepare input
            input_array = np.array([input_values]).reshape(1, -1)
            
            # Make prediction
            prediction, probability = model.predict(input_array)
            
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
