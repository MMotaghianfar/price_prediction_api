from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model and scaler
model = joblib.load('price_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return "Welcome to the Price Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    # Parse the input data from the JSON request
    data = request.get_json(force=True)
    input_data = np.array(data['data']).reshape(1, -1)

    # Scale the input data using the loaded scaler
    input_data = scaler.transform(input_data)

    # Make prediction using the loaded model
    prediction = model.predict(input_data)
    
    # Return the prediction as JSON
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
