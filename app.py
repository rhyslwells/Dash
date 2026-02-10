"""Main application entry point"""

import dash
import dash_bootstrap_components as dbc
from data import load_data, prepare_data
from model import PovertyModel
from layout import create_layout
from callbacks import register_callbacks


# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Load and prepare data
_, poverty = load_data()
df_model, features, gini_col = prepare_data(poverty)

# Train model
model = PovertyModel()
model.train(df_model, features)

# Create layout
app.layout = create_layout(model, df_model, gini_col, features)

# Register callbacks
register_callbacks(app, model)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=10000)
