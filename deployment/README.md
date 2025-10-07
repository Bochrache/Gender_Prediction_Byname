# Gender Prediction Model Deployment

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
