import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.model_selection import cross_val_score, learning_curve, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

# Set the style and color palette
sns.set_style("whitegrid")
gender_palette = ["#ADD8E6", "#FFB6C1"]  # Blue for male (1), Pink for female (2)

# Set higher DPI for better image quality
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Create output directory for cross-validation results
os.makedirs('models/cross_validation', exist_ok=True)

# Load the training and testing data
print("Loading training and testing data...")
with open('models/train_test_data.pkl', 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']

# Sample a smaller subset for cross-validation to improve performance
print("Sampling a smaller subset for cross-validation...")
from sklearn.model_selection import train_test_split
X_train_sample, _, y_train_sample, _ = train_test_split(
    X_train, y_train, train_size=0.2, random_state=42, stratify=y_train
)
print(f"Original training set shape: {X_train.shape}")
print(f"Sampled training set shape: {X_train_sample.shape}")

# Load the trained models
print("\nLoading trained models...")
models = {}

try:
    with open('models/logistic_regression.pkl', 'rb') as f:
        models['Logistic_Regression'] = pickle.load(f)
    print("Loaded Logistic Regression model")
except:
    print("Logistic Regression model not found")

try:
    with open('models/xgboost.pkl', 'rb') as f:
        models['XGBoost'] = pickle.load(f)
    print("Loaded XGBoost model")
except:
    print("XGBoost model not found")

try:
    with open('models/random_forest.pkl', 'rb') as f:
        models['Random_Forest'] = pickle.load(f)
    print("Loaded Random Forest model")
except:
    print("Random Forest model not found")

# Function to perform cross-validation
def perform_cross_validation(model, X, y, model_name, n_splits=3):
    print(f"\n=== Cross-Validation for {model_name} ===")
    
    # Define the cross-validation strategy
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Perform cross-validation
    start_time = time.time()
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    end_time = time.time()
    
    # Print results
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f}")
    print(f"Standard deviation: {cv_scores.std():.4f}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    
    # Visualize cross-validation results
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, n_splits + 1), cv_scores, color='skyblue')
    plt.axhline(y=cv_scores.mean(), color='red', linestyle='--', label=f'Mean Accuracy: {cv_scores.mean():.4f}')
    plt.xlabel('Fold', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Cross-Validation Accuracy - {model_name}', fontsize=15)
    plt.xticks(range(1, n_splits + 1))
    plt.ylim(0.8, 1.0)  # Adjust as needed
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'models/cross_validation/{model_name}_cv_scores.png')
    plt.close()
    
    return {
        'model_name': model_name,
        'cv_scores': cv_scores,
        'mean_accuracy': cv_scores.mean(),
        'std_accuracy': cv_scores.std(),
        'time_taken': end_time - start_time
    }

# Function to generate learning curves
def plot_learning_curve(model, X, y, model_name, cv=3, train_sizes=np.linspace(0.1, 1.0, 5)):
    print(f"\n=== Learning Curve for {model_name} ===")
    
    # Define the cross-validation strategy
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Generate learning curve
    start_time = time.time()
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv_strategy, train_sizes=train_sizes, scoring='accuracy', n_jobs=-1
    )
    end_time = time.time()
    
    # Calculate mean and standard deviation
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Print results
    print(f"Final training score: {train_mean[-1]:.4f}")
    print(f"Final validation score: {test_mean[-1]:.4f}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.plot(train_sizes, test_mean, 'o-', color='red', label='Cross-validation score')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='red')
    plt.xlabel('Training examples', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Learning Curve - {model_name}', fontsize=15)
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'models/cross_validation/{model_name}_learning_curve.png')
    plt.close()
    
    return {
        'model_name': model_name,
        'train_sizes': train_sizes,
        'train_mean': train_mean,
        'test_mean': test_mean,
        'time_taken': end_time - start_time
    }

# Perform cross-validation and learning curve analysis for each model
cv_results = []
lc_results = []

for model_name, model in models.items():
    print(f"\nEvaluating {model_name}...")
    
    # Perform cross-validation
    cv_result = perform_cross_validation(model, X_train_sample, y_train_sample, model_name)
    cv_results.append(cv_result)
    
    # Generate learning curve
    lc_result = plot_learning_curve(model, X_train_sample, y_train_sample, model_name)
    lc_results.append(lc_result)

# Create a DataFrame for cross-validation results
cv_df = pd.DataFrame(cv_results)
cv_df = cv_df.sort_values('mean_accuracy', ascending=False)
print("\n=== Cross-Validation Results ===")
print(cv_df[['model_name', 'mean_accuracy', 'std_accuracy', 'time_taken']])

# Save cross-validation results to CSV
cv_df[['model_name', 'mean_accuracy', 'std_accuracy', 'time_taken']].to_csv('models/cross_validation/cv_results.csv', index=False)

# Visualize cross-validation comparison
plt.figure(figsize=(12, 6))
ax = sns.barplot(x='model_name', y='mean_accuracy', data=cv_df)
plt.errorbar(x=range(len(cv_df)), y=cv_df['mean_accuracy'], yerr=cv_df['std_accuracy'], fmt='none', color='black', capsize=5)
plt.xlabel('Model', fontsize=12)
plt.ylabel('Mean Accuracy', fontsize=12)
plt.title('Cross-Validation Accuracy Comparison', fontsize=15)
plt.ylim(0.8, 1.0)  # Adjust as needed
for i, row in enumerate(cv_df.itertuples()):
    ax.text(i, row.mean_accuracy + 0.01, f'{row.mean_accuracy:.4f}', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('models/cross_validation/cv_comparison.png')
plt.close()

print("\nCross-validation and learning curve analysis completed successfully!")
