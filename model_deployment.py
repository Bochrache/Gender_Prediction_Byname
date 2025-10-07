import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string
import os

# Create directories for deployment
os.makedirs('deployment', exist_ok=True)
os.makedirs('deployment/static', exist_ok=True)
os.makedirs('deployment/templates', exist_ok=True)

# Load the best model (Random Forest based on cross-validation results)
print("Loading the best model (Random Forest)...")
with open('models/random_forest.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the character vectorizer
print("Loading the character vectorizer...")
with open('models/char_vectorizer.pkl', 'rb') as f:
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

# Create a simple HTML template for the web interface
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Gender Prediction from Names</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            text-align: center;
        }
        h1 {
            color: #333;
            margin-bottom: 30px;
        }
        .input-container {
            margin-bottom: 20px;
        }
        input[type="text"] {
            padding: 10px;
            font-size: 16px;
            width: 300px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        button {
            padding: 10px 20px;
            font-size: 16px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        button:hover {
            background-color: #45a049;
        }
        .result {
            margin-top: 30px;
            padding: 20px;
            border-radius: 8px;
            display: none;
            text-align: center;
        }
        .male {
            background-color: #ADD8E6;
            border: 2px solid #1E90FF;
        }
        .female {
            background-color: #FFB6C1;
            border: 2px solid #FF69B4;
        }
        .probability {
            font-size: 24px;
            font-weight: bold;
            margin: 10px 0;
        }
        .explanation {
            margin-top: 20px;
            font-style: italic;
            color: #555;
        }
        .model-info {
            margin-top: 50px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 8px;
            text-align: left;
        }
    </style>
</head>
<body>
    <h1>Gender Prediction from Names</h1>
    
    <div class="input-container">
        <input type="text" id="name-input" placeholder="Enter a name...">
        <button onclick="predictGender()">Predict</button>
    </div>
    
    <div id="result" class="result">
        <h2>Prediction Result</h2>
        <div id="gender-text"></div>
        <div class="probability" id="probability-text"></div>
        <div class="explanation" id="explanation-text"></div>
    </div>
    
    <div class="model-info">
        <h3>About the Model</h3>
        <p>This gender prediction model was trained on a dataset of French names from various origins. It uses character-level features to predict gender with high accuracy.</p>
        <p><strong>Model Type:</strong> Random Forest</p>
        <p><strong>Accuracy:</strong> 98.3% (Cross-validation)</p>
        <p><strong>Features:</strong> Character n-grams (1-3)</p>
    </div>
    
    <script>
        function predictGender() {
            const name = document.getElementById('name-input').value.trim();
            if (!name) {
                alert('Please enter a name');
                return;
            }
            
            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ name: name }),
            })
            .then(response => response.json())
            .then(data => {
                const resultDiv = document.getElementById('result');
                resultDiv.style.display = 'block';
                
                // Remove previous classes
                resultDiv.classList.remove('male', 'female');
                
                // Add appropriate class based on gender
                if (data.gender === 'Male') {
                    resultDiv.classList.add('male');
                } else {
                    resultDiv.classList.add('female');
                }
                
                // Update text content
                document.getElementById('gender-text').textContent = `Predicted Gender: ${data.gender}`;
                document.getElementById('probability-text').textContent = `${(data.probability * 100).toFixed(2)}% Confidence`;
                
                // Generate explanation
                let explanation = '';
                if (data.probability > 0.95) {
                    explanation = `The model is very confident that "${name}" is a ${data.gender.toLowerCase()} name.`;
                } else if (data.probability > 0.8) {
                    explanation = `The model is fairly confident that "${name}" is a ${data.gender.toLowerCase()} name.`;
                } else {
                    explanation = `The model is somewhat uncertain, but predicts "${name}" is more likely a ${data.gender.toLowerCase()} name.`;
                }
                document.getElementById('explanation-text').textContent = explanation;
            })
            .catch(error => {
                console.error('Error:', error);
                alert('An error occurred while making the prediction');
            });
        }
        
        // Allow pressing Enter to submit
        document.getElementById('name-input').addEventListener('keypress', function(event) {
            if (event.key === 'Enter') {
                predictGender();
            }
        });
    </script>
</body>
</html>
"""

# Save the HTML template to a file
with open('deployment/templates/index.html', 'w') as f:
    f.write(html_template)

# Create a Flask application
app = Flask(__name__, 
            template_folder='deployment/templates',
            static_folder='deployment/static')

@app.route('/')
def home():
    return render_template_string(html_template)

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

# Save the Flask app to a file
with open('deployment/app.py', 'w') as f:
    f.write("""
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

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
    app.run(host='0.0.0.0', port=5000, debug=True)
""")

# Create a requirements.txt file for deployment
with open('deployment/requirements.txt', 'w') as f:
    f.write("""
flask==2.0.1
numpy==1.21.0
pandas==1.3.0
scikit-learn==1.0.0
""")

# Create a README.md file with instructions
with open('deployment/README.md', 'w') as f:
    f.write("""# Gender Prediction Model Deployment

This is a web application that predicts gender based on names using a machine learning model.

## Setup and Running

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the application:
   ```
   python app.py
   ```

3. Open a web browser and navigate to:
   ```
   http://localhost:5000
   ```

## Model Information

- **Model Type:** Random Forest
- **Accuracy:** 98.3% (Cross-validation)
- **Features:** Character n-grams (1-3)
- **Training Data:** French names dataset with various origins

## API Usage

You can also use the API endpoint directly:

```
POST /predict
Content-Type: application/json

{
    "name": "Jean"
}
```

Response:
```json
{
    "name": "Jean",
    "gender": "Male",
    "probability": 0.95,
    "gender_code": 1
}
```
""")

# Test the prediction function
test_names = ['Jean', 'Marie', 'Alex', 'Sophie', 'Mohammed', 'Fatima']
print("\nTesting prediction function with sample names:")
for name in test_names:
    result = predict_gender(name)
    print(f"{name}: {result['gender']} ({result['probability']:.4f})")

print("\nDeployment files created successfully!")
print("To run the application, navigate to the deployment directory and run:")
print("  python app.py")
print("Then open a web browser and go to: http://localhost:5000")

import webbrowser
import threading

def open_browser():
    webbrowser.open_new("http://localhost:5000")

if __name__ == "__main__":
    threading.Timer(1.5, open_browser).start()
    app.run(debug=True)

# Run the Flask app if this script is executed directly
if __name__ == '__main__':
    print("\nStarting the Flask application...")
    app.run(host='0.0.0.0', port=5000, debug=True)
