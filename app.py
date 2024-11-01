from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load your model
model = joblib.load('price_prediction_model.pkl')

@app.route('/')
def home():
    return "Welcome to the Price Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    # Parse the input data
    data = request.get_json(force=True)
    input_data = np.array(data['input']).reshape(1, -1)

    # Scale the input data using the same scaler used for training
    scaler = StandardScaler()
    input_data = scaler.fit_transform(input_data)

    # Make prediction
    prediction = model.predict(input_data)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
