from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load your model and scaler
model = joblib.load('price_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')  # Assuming you saved your scaler during training

@app.route('/')
def home():
    return "Welcome to the Price Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    # Parse the input data
    data = request.get_json(force=True)
    input_data = np.array(data['data']).reshape(1, -1)

    # Scale the input data using the same scaler used for training
    input_data = scaler.transform(input_data)  # Use transform instead of fit_transform

    # Make prediction
    prediction = model.predict(input_data)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
