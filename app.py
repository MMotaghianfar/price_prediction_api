from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load your model
model = joblib.load('price_prediction_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = [
        float(request.form['avg_area_income']),
        float(request.form['avg_area_house_age']),
        float(request.form['avg_area_num_rooms']),
        float(request.form['avg_area_num_bedrooms']),
        float(request.form['area_population'])
    ]
    input_data = np.array(input_data).reshape(1, -1)

    # Scale the input data using the same scaler used for training
    scaler = StandardScaler()
    input_data = scaler.fit_transform(input_data)

    # Make prediction
    prediction = model.predict(input_data)
    return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
