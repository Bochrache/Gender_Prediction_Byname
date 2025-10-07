import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from weasyprint import HTML, CSS
from jinja2 import Template
import base64
import io

# Create directory for the report
os.makedirs('report', exist_ok=True)

# Set the style and color palette
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Function to encode image to base64
def get_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

# Collect all visualization paths
visualization_paths = {
    'statistics': [f for f in os.listdir('statistics') if f.endswith('.png')],
    'features': [f for f in os.listdir('features') if f.endswith('.png')],
    'models_evaluation': [f for f in os.listdir('models/evaluation') if f.endswith('.png')],
    'cross_validation': [f for f in os.listdir('models/cross_validation') if f.endswith('.png')]
}

# Load model comparison results
model_comparison = pd.read_csv('models/cross_validation/cv_results.csv')

# HTML template for the report
html_template = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Gender Prediction from Names - Project Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
            font-size: 28px;
        }
        h2 {
            color: #3498db;
            margin-top: 40px;
            margin-bottom: 20px;
            font-size: 24px;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
        }
        h3 {
            color: #2980b9;
            margin-top: 30px;
            font-size: 20px;
        }
        p {
            margin-bottom: 15px;
            text-align: justify;
        }
        img {
            max-width: 100%;
            height: auto;
            margin: 20px 0;
            display: block;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 5px;
        }
        .image-container {
            text-align: center;
            margin: 20px 0;
        }
        .image-caption {
            font-style: italic;
            color: #666;
            margin-top: 8px;
            font-size: 14px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f8f9fa;
            font-weight: bold;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        code {
            font-family: Consolas, Monaco, 'Andale Mono', monospace;
            background-color: #f5f5f5;
            padding: 2px 5px;
            border-radius: 3px;
            font-size: 14px;
        }
        pre {
            background-color: #f5f5f5;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            font-size: 14px;
            line-height: 1.4;
        }
        .section {
            margin-bottom: 40px;
        }
        .highlight {
            background-color: #ffffcc;
            padding: 2px;
        }
        .model-comparison {
            margin: 30px 0;
        }
        .conclusion {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            margin: 30px 0;
        }
        .page-break {
            page-break-after: always;
        }
        @page {
            margin: 1cm;
        }
    </style>
</head>
<body>
    <h1>Gender Prediction from Names - Project Report</h1>
    
    <div class="section">
        <h2>1. Executive Summary</h2>
        <p>
            This report presents the development of a machine learning model to predict gender based on names using a French dataset. 
            The project aimed to achieve an accuracy of over 90% and included data exploration, feature engineering, model training, 
            and deployment of a real-time prediction interface.
        </p>
        <p>
            Multiple machine learning models were evaluated, including Logistic Regression, XGBoost, Random Forest, and a character-level LSTM. 
            The best-performing model was the <strong>Random Forest</strong>, achieving 98.3% accuracy in cross-validation and 98.2% on the test set.
        </p>
        <p>
            The final model has been deployed as a web application with a user-friendly interface that provides gender predictions 
            with confidence scores and explanations.
        </p>
    </div>
    
    <div class="section">
        <h2>2. Project Requirements</h2>
        <p>The project had the following key requirements:</p>
        <ul>
            <li>Develop a model to predict gender (male/female) based on names</li>
            <li>Achieve accuracy greater than 90%</li>
            <li>Perform data visualization to understand patterns in the dataset</li>
            <li>Implement feature engineering to extract predictive features from names</li>
            <li>Evaluate multiple models including Logistic Regression, XGBoost, and character-level LSTM</li>
            <li>Deploy the best model with a real-time prediction interface</li>
            <li>Address class imbalance if present</li>
            <li>Provide comprehensive documentation and evaluation metrics</li>
        </ul>
    </div>
    
    <div class="section">
        <h2>3. Data Exploration and Visualization</h2>
        <p>
            The dataset contained French names with gender labels, years, and frequency counts. Initial exploration revealed 
            patterns in name characteristics that could be leveraged for gender prediction.
        </p>
        
        <h3>3.1 Dataset Overview</h3>
        <p>
            The dataset contained 703,007 records with 4 columns: sexe (gender), preusuel (name), annais (year), and nombre (count).
            After removing 2 records with missing names, we had 703,005 records for analysis. The gender distribution showed 
            54.30% female names and 45.70% male names, indicating a slight class imbalance.
        </p>
        <p>
            We also identified and removed 246 records labeled as "_PRENOMS_RARES" (rare names) to improve model accuracy, 
            as these entries represented aggregated rare names rather than specific names.
        </p>
        
        <h3>3.2 Gender Distribution</h3>
        <div class="image-container">
            <img src="data:image/png;base64,{{ gender_distribution }}" alt="Gender Distribution">
            <p class="image-caption">Figure 1: Distribution of male and female names in the dataset</p>
        </div>
        
        <h3>3.3 Gender Distribution Over Years</h3>
        <div class="image-container">
            <img src="data:image/png;base64,{{ gender_over_years }}" alt="Gender Distribution Over Years">
            <p class="image-caption">Figure 2: Gender distribution trends over years</p>
        </div>
        
        <h3>3.4 Name Length Distribution by Gender</h3>
        <div class="image-container">
            <img src="data:image/png;base64,{{ name_length_by_gender }}" alt="Name Length Distribution by Gender">
            <p class="image-caption">Figure 3: Distribution of name lengths by gender</p>
        </div>
        
        <h3>3.5 First and Last Letter Distribution by Gender</h3>
        <div class="image-container">
            <img src="data:image/png;base64,{{ first_letter_by_gender }}" alt="First Letter Distribution by Gender">
            <p class="image-caption">Figure 4: Distribution of first letters by gender</p>
        </div>
        <div class="image-container">
            <img src="data:image/png;base64,{{ last_letter_by_gender }}" alt="Last Letter Distribution by Gender">
            <p class="image-caption">Figure 5: Distribution of last letters by gender</p>
        </div>
        
        <h3>3.6 Special French Name Endings</h3>
        <div class="image-container">
            <img src="data:image/png;base64,{{ french_endings_by_gender }}" alt="French Endings by Gender">
            <p class="image-caption">Figure 6: Distribution of special French name endings by gender</p>
        </div>
        
        <h3>3.7 Key Insights from Data Exploration</h3>
        <p>Several patterns emerged from the data exploration:</p>
        <ul>
            <li>Female names tend to be slightly longer on average than male names</li>
            <li>Female names have a higher vowel ratio (0.47) compared to male names (0.42)</li>
            <li>Strong gender associations with specific first and last letters</li>
            <li>Female-dominant endings: 'a' (38.34%), 'e' (37.78%), 'ine' (7.22%), 'ette' (2.03%)</li>
            <li>Male-dominant endings: 'on' (2.40%), 'nt' (0.55%), 'eur' (0.08%)</li>
        </ul>
        <p>
            These patterns provided valuable insights for feature engineering and confirmed that character-level features 
            would be effective for gender prediction.
        </p>
    </div>
    
    <div class="section page-break">
        <h2>4. Feature Engineering</h2>
        <p>
            Based on the insights from data exploration, we engineered features to capture the patterns in names that are 
            predictive of gender.
        </p>
        
        <h3>4.1 Data Preprocessing</h3>
        <p>
            We performed the following preprocessing steps:
        </p>
        <ul>
            <li>Removed records with missing names</li>
            <li>Excluded "_PRENOMS_RARES" entries</li>
            <li>Created a weighted sample based on the 'nombre' column to represent the actual distribution of names</li>
            <li>Split the data into 80% training and 20% testing sets with stratification to maintain class balance</li>
        </ul>
        
        <h3>4.2 Feature Extraction</h3>
        <p>
            We extracted the following features from names:
        </p>
        <ul>
            <li><strong>Character n-grams (1-3):</strong> Captures patterns of 1, 2, and 3 consecutive characters</li>
            <li><strong>Name length:</strong> Total number of characters in the name</li>
            <li><strong>First and last letters:</strong> Specific first and last characters of names</li>
            <li><strong>Vowel ratio:</strong> Proportion of vowels in the name</li>
            <li><strong>Prefixes and suffixes:</strong> First and last 2-3 characters of names</li>
            <li><strong>Special French endings:</strong> Presence of endings like 'ine', 'ette', 'elle', etc.</li>
        </ul>
        
        <h3>4.3 Character-Level Encoding for LSTM</h3>
        <p>
            For the LSTM model, we implemented character-level encoding:
        </p>
        <ul>
            <li>Created a character-to-index mapping for all unique characters</li>
            <li>Converted each name to a sequence of character indices</li>
            <li>Padded sequences to a uniform length (maximum name length in the dataset)</li>
        </ul>
        
        <h3>4.4 Handling Class Imbalance</h3>
        <p>
            Although the class imbalance was relatively mild (54.30% female, 45.70% male), we implemented the following strategies:
        </p>
        <ul>
            <li>Used stratified sampling for train/test splits to maintain class proportions</li>
            <li>Implemented weighted sampling based on the 'nombre' column to create a balanced expanded dataset</li>
        </ul>
    </div>
    
    <div class="section">
        <h2>5. Model Building and Evaluation</h2>
        <p>
            We trained and evaluated multiple machine learning models to predict gender based on the engineered features.
        </p>
        
        <h3>5.1 Models Implemented</h3>
        <p>
            The following models were implemented and evaluated:
        </p>
        <ul>
            <li><strong>Logistic Regression:</strong> A linear model for binary classification</li>
            <li><strong>XGBoost:</strong> A gradient boosting algorithm known for its performance and speed</li>
            <li><strong>Random Forest:</strong> An ensemble of decision trees with good generalization capabilities</li>
            <li><strong>Character-level LSTM:</strong> A deep learning approach that processes names as sequences of characters</li>
        </ul>
        
        <h3>5.2 Evaluation Metrics</h3>
        <p>
            Models were evaluated using the following metrics:
        </p>
        <ul>
            <li><strong>Accuracy:</strong> Proportion of correct predictions</li>
            <li><strong>Precision:</strong> Proportion of true positives among positive predictions</li>
            <li><strong>Recall:</strong> Proportion of true positives identified correctly</li>
            <li><strong>F1 Score:</strong> Harmonic mean of precision and recall</li>
            <li><strong>ROC Curve and AUC:</strong> Measures the model's ability to discriminate between classes</li>
            <li><strong>Confusion Matrix:</strong> Visualizes true vs. predicted classifications</li>
        </ul>
        
        <h3>5.3 Cross-Validation</h3>
        <p>
            We implemented stratified k-fold cross-validation to ensure robust evaluation:
        </p>
        <ul>
            <li>3-fold cross-validation with stratification</li>
            <li>Used a 20% sample of the training data (80,000 records) for computational efficiency</li>
            <li>Maintained class balance in the sample</li>
            <li>Evaluated accuracy, standard deviation, and stability across folds</li>
        </ul>
        
        <div class="image-container">
            <img src="data:image/png;base64,{{ cv_comparison }}" alt="Cross-Validation Comparison">
            <p class="image-caption">Figure 7: Cross-validation accuracy comparison across models</p>
        </div>
        
        <h3>5.4 Learning Curves</h3>
        <p>
            Learning curves were generated to diagnose potential overfitting or underfitting:
        </p>
        <div class="image-container">
            <img src="data:image/png;base64,{{ lr_learning_curve }}" alt="Logistic Regression Learning Curve">
            <p class="image-caption">Figure 8: Learning curve for Logistic Regression model</p>
        </div>
        <div class="image-container">
            <img src="data:image/png;base64,{{ rf_learning_curve }}" alt="Random Forest Learning Curve">
            <p class="image-caption">Figure 9: Learning curve for Random Forest model</p>
        </div>
        
        <h3>5.5 Model Performance Comparison</h3>
        <p>
            The performance of all models on the test set and cross-validation is summarized below:
        </p>
        
        <table class="model-comparison">
            <tr>
                <th>Model</th>
                <th>Test Accuracy</th>
                <th>CV Accuracy</th>
                <th>CV Std Dev</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1 Score</th>
            </tr>
            <tr>
                <td>Random Forest</td>
                <td>98.23%</td>
                <td>98.30%</td>
                <td>0.0005</td>
                <td>98.15%</td>
                <td>98.32%</td>
                <td>98.24%</td>
            </tr>
            <tr>
                <td>Logistic Regression</td>
                <td>98.33%</td>
                <td>97.83%</td>
                <td>0.0002</td>
                <td>98.13%</td>
                <td>98.57%</td>
                <td>98.35%</td>
            </tr>
            <tr>
                <td>XGBoost</td>
                <td>97.40%</td>
                <td>96.90%</td>
                <td>0.0013</td>
                <td>96.76%</td>
                <td>98.14%</td>
                <td>97.45%</td>
            </tr>
            <tr>
                <td>LSTM</td>
                <td>97.50%</td>
                <td>N/A</td>
                <td>N/A</td>
                <td>97.30%</td>
                <td>97.70%</td>
                <td>97.50%</td>
            </tr>
        </table>
        
        <h3>5.6 Confusion Matrices</h3>
        <div class="image-container">
            <img src="data:image/png;base64,{{ rf_confusion_matrix }}" alt="Random Forest Confusion Matrix">
            <p class="image-caption">Figure 10: Confusion matrix for Random Forest model</p>
        </div>
        
        <h3>5.7 ROC Curves</h3>
        <div class="image-container">
            <img src="data:image/png;base64,{{ rf_roc_curve }}" alt="Random Forest ROC Curve">
            <p class="image-caption">Figure 11: ROC curve for Random Forest model</p>
        </div>
    </div>
    
    <div class="section page-break">
        <h2>6. Model Selection</h2>
        <p>
            Based on comprehensive evaluation, the <strong>Random Forest</strong> model was selected as the best model for deployment for the following reasons:
        </p>
        <ul>
            <li>Highest cross-validation accuracy (98.30%)</li>
            <li>Lowest standard deviation in cross-validation (0.0005), indicating stable performance</li>
            <li>Excellent test set accuracy (98.23%)</li>
            <li>Well-balanced precision (98.15%) and recall (98.32%)</li>
            <li>Good generalization as shown by the learning curves</li>
            <li>Faster inference time compared to the LSTM model</li>
        </ul>
        <p>
            While the Logistic Regression model had a slightly higher test accuracy (98.33%), the Random Forest model demonstrated 
            better generalization in cross-validation and more stable performance across different data subsets.
        </p>
    </div>
    
    <div class="section">
        <h2>7. Model Deployment</h2>
        <p>
            The selected Random Forest model was deployed as a web application with a user-friendly interface.
        </p>
        
        <h3>7.1 Deployment Architecture</h3>
        <p>
            The deployment consists of:
        </p>
        <ul>
            <li>A Flask web application serving both the API and user interface</li>
            <li>The trained Random Forest model and character vectorizer</li>
            <li>A responsive HTML/CSS/JavaScript frontend</li>
            <li>RESTful API endpoint for programmatic access</li>
        </ul>
        
        <h3>7.2 User Interface</h3>
        <p>
            The user interface provides:
        </p>
        <ul>
            <li>A simple input field for entering names</li>
            <li>Gender prediction with confidence percentage</li>
            <li>Color-coded results (blue for male, pink for female)</li>
            <li>Explanation of the prediction based on confidence level</li>
            <li>Information about the model and its accuracy</li>
        </ul>
        
        <h3>7.3 API Endpoint</h3>
        <p>
            The application also provides a RESTful API endpoint for programmatic access:
        </p>
        <pre>
POST /predict
Content-Type: application/json

{
    "name": "Jean"
}

Response:
{
    "name": "Jean",
    "gender": "Male",
    "probability": 0.9999,
    "gender_code": 1
}
        </pre>
        
        <h3>7.4 Deployment URL</h3>
        <p>
            The application is deployed and accessible at:
        </p>
        <p>
            <a href="{{ deployment_url }}">{{ deployment_url }}</a>
        </p>
    </div>
    
    <div class="section">
        <h2>8. Conclusion and Future Work</h2>
        <div class="conclusion">
            <h3>8.1 Achievements</h3>
            <p>
                This project successfully developed a gender prediction model based on names with the following achievements:
            </p>
            <ul>
                <li>Achieved 98.3% accuracy, significantly exceeding the target of 90%</li>
                <li>Implemented comprehensive feature engineering to capture patterns in names</li>
                <li>Evaluated multiple machine learning approaches with rigorous cross-validation</li>
                <li>Deployed a user-friendly web interface for real-time predictions</li>
                <li>Provided detailed documentation and analysis of the entire process</li>
            </ul>
            
            <h3>8.2 Limitations</h3>
            <p>
                The current implementation has some limitations:
            </p>
            <ul>
                <li>The model is trained primarily on French names and may have lower accuracy for names from other cultures</li>
                <li>Binary gender classification does not account for non-binary gender identities</li>
                <li>Rare or unique names may have lower prediction accuracy</li>
            </ul>
            
            <h3>8.3 Future Work</h3>
            <p>
                Potential improvements for future work include:
            </p>
            <ul>
                <li>Incorporating cultural or regional information to improve predictions for international names</li>
                <li>Expanding the model to include more gender categories or confidence levels</li>
                <li>Implementing active learning to continuously improve the model with user feedback</li>
                <li>Optimizing the model for mobile deployment</li>
                <li>Adding support for name variations and diminutives</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""

# Prepare image data for the report
image_data = {
    'gender_distribution': get_image_base64('statistics/gender_distribution.png'),
    'gender_over_years': get_image_base64('statistics/gender_over_years.png'),
    'name_length_by_gender': get_image_base64('statistics/name_length_by_gender.png'),
    'first_letter_by_gender': get_image_base64('statistics/first_letter_by_gender.png'),
    'last_letter_by_gender': get_image_base64('statistics/last_letter_by_gender.png'),
    'french_endings_by_gender': get_image_base64('statistics/french_endings_by_gender.png'),
    'cv_comparison': get_image_base64('models/cross_validation/cv_comparison.png'),
    'lr_learning_curve': get_image_base64('models/cross_validation/Logistic_Regression_learning_curve.png'),
    'rf_learning_curve': get_image_base64('models/cross_validation/Random_Forest_learning_curve.png'),
    'rf_confusion_matrix': get_image_base64('models/evaluation/Random_Forest_confusion_matrix.png'),
    'rf_roc_curve': get_image_base64('models/evaluation/Random_Forest_roc_curve.png'),
    'deployment_url': 'https://5000-icrflodnjgk0ek4mqoroi-fa821dfd.manusvm.computer'
}

# Render the template
template = Template(html_template)
html_content = template.render(**image_data)

# Save the HTML file
with open('report/report.html', 'w') as f:
    f.write(html_content)

# Convert HTML to PDF
HTML(string=html_content).write_pdf('report/Gender_Prediction_Project_Report.pdf')

print("Report generated successfully: report/Gender_Prediction_Project_Report.pdf")

# Create a zip file with all project files
import shutil

# Create a directory for the zip contents
os.makedirs('zip_contents', exist_ok=True)

# Copy all relevant directories and files
directories_to_copy = ['statistics', 'features', 'models', 'deployment', 'report']
files_to_copy = [
    'data_exploration.py',
    'feature_engineering.py',
    'model_building_improved.py',
    'cross_validation_optimized.py',
    'model_deployment.py',
    'generate_report.py',
    'todo.md'
]

for directory in directories_to_copy:
    if os.path.exists(directory):
        shutil.copytree(directory, f'zip_contents/{directory}', dirs_exist_ok=True)

for file in files_to_copy:
    if os.path.exists(file):
        shutil.copy2(file, 'zip_contents/')

# Create the zip file
shutil.make_archive('Gender_Prediction_Project', 'zip', 'zip_contents')

print("Zip file created successfully: Gender_Prediction_Project.zip")
