from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)


with open('random_forest.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return "Random Forest API is running"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
