from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load your model
model = joblib.load('price_prediction_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict([data['input']])
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
