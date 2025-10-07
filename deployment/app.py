import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
import webbrowser
import threading

# Load the model and vectorizer
with open('../models/random_forest.pkl', 'rb') as f:
    model = pickle.load(f)

with open('../models/char_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Create a function to preprocess and predict
def predict_gender(name):
    # Vectorize the name
    name_vec = vectorizer.transform([name])
    
    # Make prediction
    prediction = model.predict(name_vec)[0]
    probability = model.predict_proba(name_vec)[0]
    
    # Get the probability of the predicted class
    pred_probability = probability[1] if prediction == 1 else probability[0]
    
    # Return prediction (1 for male, 0 for female) and probability
    return {
        'name': name,
        'gender': 'Male' if prediction == 1 else 'Female',
        'probability': float(pred_probability),
        'gender_code': int(prediction)
    }

# Create a Flask application
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the name from the request
    data = request.get_json()
    name = data.get('name', '')
    
    if not name:
        return jsonify({'error': 'No name provided'}), 400
    
    # Make prediction
    result = predict_gender(name)
    
    return jsonify(result)

if __name__ == '__main__':

    threading.Timer(1.0, lambda: webbrowser.open('http://127.0.0.1:5001')).start()
    app.run(host='0.0.0.0', port=5001, debug=True)