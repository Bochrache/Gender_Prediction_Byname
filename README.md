# Gender Prediction by Name

A small research/project codebase for predicting the likely gender of a person from their given name using classical ML and sequence models (logistic regression, random forest, XGBoost, and a small LSTM).

Contents
- data and preprocessing scripts
- feature extraction and engineered features in `feature_engineering.py`
- model training and evaluation in `model_building_improved.py` and `cross_validation_optimized.py`
- pre-trained models stored in `models/` (not tracked if large)
- a simple Flask app in `deployment/app.py` for serving predictions

Quick start
1. Create and activate a Python virtual environment:

   python3 -m venv env
   source env/bin/activate

2. Install dependencies (the lightweight app requirements are in `deployment/requirements.txt`):

   pip install -r deployment/requirements.txt

3. Run the example web app to try predictions locally:

   python deployment/app.py

Or use the scripts directly to train/evaluate models. See the top-level .py files for usage examples.

Notes
- Large datasets and model artifacts are excluded from version control via `.gitignore`.
- If you need to track models or large data, consider Git LFS or an external storage bucket.

Author
- Repository owner: Bochrache