import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Set the style and color palette
sns.set_style("whitegrid")
gender_palette = ["#ADD8E6", "#FFB6C1"]  # Blue for male (1), Pink for female (2)

# Set higher DPI for better image quality
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Create output directory for models
os.makedirs('models', exist_ok=True)
os.makedirs('models/evaluation', exist_ok=True)

# Load the dataset and filter out "_PRENOMS_RARES"
print("Loading and filtering dataset...")
df = pd.read_csv(r"C:\Users\bochra.chemam\Downloads\Gender_Prediction_Project\nat2022.csv", sep=';')

# Count before filtering
total_records_before = len(df)
total_rare_records = df[df['preusuel'] == '_PRENOMS_RARES']['nombre'].sum()
print(f"Total records before filtering: {total_records_before:,}")
print(f"Total '_PRENOMS_RARES' records: {df[df['preusuel'] == '_PRENOMS_RARES'].shape[0]:,}")
print(f"Total individuals with rare names: {total_rare_records:,}")

# Filter out "_PRENOMS_RARES"
df_filtered = df[df['preusuel'] != '_PRENOMS_RARES'].copy()
print(f"Records after filtering: {len(df_filtered):,}")
print(f"Removed {total_records_before - len(df_filtered):,} records with '_PRENOMS_RARES'")

# Check class distribution after filtering
gender_counts = df_filtered['sexe'].value_counts()
print("\nGender distribution after filtering:")
print(gender_counts)
print(f"Male percentage: {gender_counts[1]/len(df_filtered)*100:.2f}%")
print(f"Female percentage: {gender_counts[2]/len(df_filtered)*100:.2f}%")

# Visualize the gender distribution
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=gender_counts.index, y=gender_counts.values, palette=gender_palette)
plt.title('Gender Distribution', fontsize=15)
plt.xlabel('Gender (1: Male, 2: Female)', fontsize=12)
plt.ylabel('Count', fontsize=12)
for i, v in enumerate(gender_counts.values):
    ax.text(i, v + 0.1, f'{v:,}', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('models/evaluation/filtered_gender_distribution.png')
plt.close()

# Create a new dataframe with only the necessary columns
model_df = df_filtered[['sexe', 'preusuel', 'nombre']].copy()

# Expand the dataset based on 'nombre' (count)
print("\nExpanding dataset based on name counts...")
# To avoid memory issues, we'll sample a portion of the data
# First, calculate total counts
total_count = model_df['nombre'].sum()
print(f"Total count in filtered dataset: {total_count:,}")

# Set a reasonable sample size
sample_size = 500000  # Reduced sample size to manage memory better

# Create a weighted sampling approach
print("Creating weighted sample...")
# Create a probability distribution based on nombre
model_df['sampling_weight'] = model_df['nombre'] / model_df['nombre'].sum()

# Sample with replacement to ensure we get the desired sample size
sampled_indices = np.random.choice(
    model_df.index, 
    size=sample_size, 
    replace=True, 
    p=model_df['sampling_weight']
)
sampled_df = model_df.loc[sampled_indices].reset_index(drop=True)

# Create expanded dataset
expanded_df = pd.DataFrame({
    'sexe': sampled_df['sexe'].values,
    'preusuel': sampled_df['preusuel'].values
})

print(f"Expanded dataset shape: {expanded_df.shape}")

# Check class distribution in expanded dataset
print("\nClass distribution in expanded dataset:")
expanded_gender_counts = expanded_df['sexe'].value_counts()
print(expanded_gender_counts)
print(f"Male percentage: {expanded_gender_counts[1]/len(expanded_df)*100:.2f}%")
print(f"Female percentage: {expanded_gender_counts[2]/len(expanded_df)*100:.2f}%")

# Visualize the expanded class distribution
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=expanded_gender_counts.index, y=expanded_gender_counts.values, palette=gender_palette)
plt.title('Gender Distribution in Expanded Dataset', fontsize=15)
plt.xlabel('Gender (1: Male, 2: Female)', fontsize=12)
plt.ylabel('Count', fontsize=12)
for i, v in enumerate(expanded_gender_counts.values):
    ax.text(i, v + 0.1, f'{v:,}', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('models/evaluation/expanded_gender_distribution.png')
plt.close()

# Feature extraction for character-level analysis
print("\nExtracting character-level features...")

# 1. Character n-grams using CountVectorizer
print("Generating character n-grams...")
from sklearn.feature_extraction.text import CountVectorizer
char_vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 3))
X_char_ngrams = char_vectorizer.fit_transform(expanded_df['preusuel'])
char_feature_names = char_vectorizer.get_feature_names_out()
print(f"Number of character n-gram features: {len(char_feature_names)}")

# Save the vectorizer for later use in deployment
with open('models/char_vectorizer.pkl', 'wb') as f:
    pickle.dump(char_vectorizer, f)

# 2. One-hot encoding for character-level LSTM
print("Preparing character-level encoding for LSTM...")
# Get all unique characters
all_chars = set()
for name in expanded_df['preusuel']:
    all_chars.update(name.lower())
char_to_idx = {c: i+1 for i, c in enumerate(sorted(all_chars))}  # 0 reserved for padding
idx_to_char = {i+1: c for i, c in enumerate(sorted(all_chars))}
print(f"Number of unique characters: {len(all_chars)}")

# Save character mappings for later use
with open('models/char_mappings.pkl', 'wb') as f:
    pickle.dump({'char_to_idx': char_to_idx, 'idx_to_char': idx_to_char}, f)

# Prepare train/test split for modeling
print("\nPreparing train/test split...")
from sklearn.model_selection import train_test_split
X = X_char_ngrams
y = expanded_df['sexe'].values
# Convert to binary classification (1: Male, 0: Female)
y_binary = (y == 1).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42, stratify=y_binary)
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Check class distribution in train and test sets
print("\nClass distribution in training set:")
train_class_counts = pd.Series(y_train).value_counts()
print(train_class_counts)
print(f"Male percentage: {train_class_counts[1]/len(y_train)*100:.2f}%")
print(f"Female percentage: {train_class_counts[0]/len(y_train)*100:.2f}%")

print("\nClass distribution in testing set:")
test_class_counts = pd.Series(y_test).value_counts()
print(test_class_counts)
print(f"Male percentage: {test_class_counts[1]/len(y_test)*100:.2f}%")
print(f"Female percentage: {test_class_counts[0]/len(y_test)*100:.2f}%")

# Handle class imbalance if necessary
imbalance_ratio = expanded_gender_counts.max() / expanded_gender_counts.min()
if imbalance_ratio > 1.5:
    print("\nHandling class imbalance...")
    # Option 1: Class weights (will be used in model training)
    # Calculate class weights
    class_weights = {0: len(y_train) / (2 * (len(y_train) - sum(y_train))),
                     1: len(y_train) / (2 * sum(y_train))}
    print(f"Class weights: {class_weights}")
    
    # Option 2: SMOTE (Synthetic Minority Over-sampling Technique)
    print("Applying SMOTE to balance classes...")
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"Original training set shape: {X_train.shape}")
    print(f"SMOTE-balanced training set shape: {X_train_smote.shape}")
    
    # Check class distribution after SMOTE
    print("\nClass distribution after SMOTE:")
    smote_class_counts = pd.Series(y_train_smote).value_counts()
    print(smote_class_counts)
    print(f"Male percentage: {smote_class_counts[1]/len(y_train_smote)*100:.2f}%")
    print(f"Female percentage: {smote_class_counts[0]/len(y_train_smote)*100:.2f}%")
    
    # Save both original and SMOTE-balanced datasets
    with open('models/train_test_data.pkl', 'wb') as f:
        pickle.dump({
            'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test,
            'X_train_smote': X_train_smote, 'y_train_smote': y_train_smote,
            'class_weights': class_weights
        }, f)
else:
    # Save original datasets
    with open('models/train_test_data.pkl', 'wb') as f:
        pickle.dump({
            'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test
        }, f)
    X_train_smote = None
    y_train_smote = None

# Function to evaluate and visualize model performance
def evaluate_model(model, X_test, y_test, model_name):
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Print classification report
    print(f"\n=== {model_name} Performance ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Female', 'Male']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Female', 'Male'], 
                yticklabels=['Female', 'Male'])
    plt.title(f'Confusion Matrix - {model_name}', fontsize=15)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'models/evaluation/{model_name}_confusion_matrix.png')
    plt.close()
    
    # ROC Curve (if probability predictions are available)
    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {model_name}', fontsize=15)
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'models/evaluation/{model_name}_roc_curve.png')
        plt.close()
        
        # Precision-Recall Curve
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall_curve, precision_curve)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall_curve, precision_curve, color='green', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve - {model_name}', fontsize=15)
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'models/evaluation/{model_name}_pr_curve.png')
        plt.close()
    
    # Return metrics for comparison
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# 1. Logistic Regression Model
print("\n=== Training Logistic Regression Model ===")
# Train on original data
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

# Save the model
with open('models/logistic_regression.pkl', 'wb') as f:
    pickle.dump(lr_model, f)

# Evaluate the model
lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic_Regression")

# Train on SMOTE-balanced data if available
if X_train_smote is not None:
    print("\n=== Training Logistic Regression Model with SMOTE-balanced data ===")
    lr_model_smote = LogisticRegression(max_iter=1000, random_state=42)
    lr_model_smote.fit(X_train_smote, y_train_smote)
    
    # Save the model
    with open('models/logistic_regression_smote.pkl', 'wb') as f:
        pickle.dump(lr_model_smote, f)
    
    # Evaluate the model
    lr_smote_metrics = evaluate_model(lr_model_smote, X_test, y_test, "Logistic_Regression_SMOTE")

# 2. XGBoost Model
print("\n=== Training XGBoost Model ===")
# Train on original data
xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Save the model
with open('models/xgboost.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

# Evaluate the model
xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")

# Train on SMOTE-balanced data if available
if X_train_smote is not None:
    print("\n=== Training XGBoost Model with SMOTE-balanced data ===")
    xgb_model_smote = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb_model_smote.fit(X_train_smote, y_train_smote)
    
    # Save the model
    with open('models/xgboost_smote.pkl', 'wb') as f:
        pickle.dump(xgb_model_smote, f)
    
    # Evaluate the model
    xgb_smote_metrics = evaluate_model(xgb_model_smote, X_test, y_test, "XGBoost_SMOTE")

# 3. Random Forest Model (as an additional model)
print("\n=== Training Random Forest Model ===")
# Train on original data
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save the model
with open('models/random_forest.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

# Evaluate the model
rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random_Forest")

# Train on SMOTE-balanced data if available
if X_train_smote is not None:
    print("\n=== Training Random Forest Model with SMOTE-balanced data ===")
    rf_model_smote = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model_smote.fit(X_train_smote, y_train_smote)
    
    # Save the model
    with open('models/random_forest_smote.pkl', 'wb') as f:
        pickle.dump(rf_model_smote, f)
    
    # Evaluate the model
    rf_smote_metrics = evaluate_model(rf_model_smote, X_test, y_test, "Random_Forest_SMOTE")

# 4. Character-level LSTM Model
print("\n=== Training Character-level LSTM Model ===")
# Prepare data for LSTM model
max_name_length = expanded_df['preusuel'].str.len().max()
print(f"Maximum name length: {max_name_length}")

# Function to convert a name to a sequence of character indices
def name_to_sequence(name, max_length):
    name = name.lower()
    sequence = [char_to_idx.get(c, 0) for c in name]
    # Pad sequence to max_length
    if len(sequence) < max_length:
        sequence = sequence + [0] * (max_length - len(sequence))
    else:
        sequence = sequence[:max_length]
    return sequence

# Convert names to sequences
X_lstm = np.array([name_to_sequence(name, max_name_length) for name in expanded_df['preusuel']])
y_lstm = y_binary

# Split the LSTM data
X_lstm_train, X_lstm_test, y_lstm_train, y_lstm_test = train_test_split(
    X_lstm, y_lstm, test_size=0.2, random_state=42, stratify=y_lstm
)

# Build the LSTM model
embedding_dim = 32
lstm_units = 64
vocab_size = len(char_to_idx) + 1  # +1 for padding

lstm_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_name_length),
    Bidirectional(LSTM(lstm_units, return_sequences=True)),
    Dropout(0.2),
    Bidirectional(LSTM(lstm_units)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

lstm_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(lstm_model.summary())

# Callbacks for training
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(
    'models/lstm_model.h5', 
    monitor='val_accuracy', 
    save_best_only=True, 
    mode='max'
)

# Train the LSTM model
history = lstm_model.fit(
    X_lstm_train, y_lstm_train,
    epochs=20,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stopping, model_checkpoint]
)

# Evaluate the LSTM model
lstm_loss, lstm_accuracy = lstm_model.evaluate(X_lstm_test, y_lstm_test)
print(f"\nLSTM Test Accuracy: {lstm_accuracy:.4f}")
print(f"LSTM Test Loss: {lstm_loss:.4f}")

# Make predictions
y_lstm_pred = (lstm_model.predict(X_lstm_test) > 0.5).astype(int).flatten()
y_lstm_pred_proba = lstm_model.predict(X_lstm_test).flatten()

# Calculate metrics
lstm_precision = precision_score(y_lstm_test, y_lstm_pred)
lstm_recall = recall_score(y_lstm_test, y_lstm_pred)
lstm_f1 = f1_score(y_lstm_test, y_lstm_pred)

print("\n=== LSTM Model Performance ===")
print(f"Accuracy: {lstm_accuracy:.4f}")
print(f"Precision: {lstm_precision:.4f}")
print(f"Recall: {lstm_recall:.4f}")
print(f"F1 Score: {lstm_f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_lstm_test, y_lstm_pred, target_names=['Female', 'Male']))

# Confusion Matrix for LSTM
cm_lstm = confusion_matrix(y_lstm_test, y_lstm_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_lstm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Female', 'Male'], 
            yticklabels=['Female', 'Male'])
plt.title('Confusion Matrix - LSTM', fontsize=15)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('models/evaluation/LSTM_confusion_matrix.png')
plt.close()

# ROC Curve for LSTM
fpr_lstm, tpr_lstm, _ = roc_curve(y_lstm_test, y_lstm_pred_proba)
roc_auc_lstm = auc(fpr_lstm, tpr_lstm)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lstm, tpr_lstm, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_lstm:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - LSTM', fontsize=15)
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig('models/evaluation/LSTM_roc_curve.png')
plt.close()

# Precision-Recall Curve for LSTM
precision_curve_lstm, recall_curve_lstm, _ = precision_recall_curve(y_lstm_test, y_lstm_pred_proba)
pr_auc_lstm = auc(recall_curve_lstm, precision_curve_lstm)

plt.figure(figsize=(8, 6))
plt.plot(recall_curve_lstm, precision_curve_lstm, color='green', lw=2, label=f'PR curve (area = {pr_auc_lstm:.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve - LSTM', fontsize=15)
plt.legend(loc="lower left")
plt.grid(True)
plt.tight_layout()
plt.savefig('models/evaluation/LSTM_pr_curve.png')
plt.close()

# Training history plot for LSTM
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('LSTM Model Accuracy', fontsize=15)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('LSTM Model Loss', fontsize=15)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('models/evaluation/LSTM_training_history.png')
plt.close()

# Add LSTM metrics to the comparison
lstm_metrics = {
    'model_name': 'LSTM',
    'accuracy': lstm_accuracy,
    'precision': lstm_precision,
    'recall': lstm_recall,
    'f1': lstm_f1
}

# Collect all metrics for comparison
all_metrics = [lr_metrics, xgb_metrics, rf_metrics, lstm_metrics]
if X_train_smote is not None:
    all_metrics.extend([lr_smote_metrics, xgb_smote_metrics, rf_smote_metrics])

# Create a DataFrame for comparison
metrics_df = pd.DataFrame(all_metrics)
metrics_df = metrics_df.sort_values('accuracy', ascending=False)
print("\n=== Model Comparison ===")
print(metrics_df)

# Save metrics to CSV
metrics_df.to_csv('models/evaluation/model_comparison.csv', index=False)

# Visualize model comparison
plt.figure(figsize=(12, 8))
metrics_df_plot = metrics_df.set_index('model_name')
metrics_df_plot[['accuracy', 'precision', 'recall', 'f1']].plot(kind='bar', figsize=(12, 6))
plt.title('Model Performance Comparison', fontsize=15)
plt.xlabel('Model', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig('models/evaluation/model_comparison.png')
plt.close()

# Identify the best model based on accuracy
best_model_row = metrics_df.iloc[0]
best_model_name = best_model_row['model_name']
best_model_accuracy = best_model_row['accuracy']
print(f"\nBest model: {best_model_name} with accuracy: {best_model_accuracy:.4f}")

# Save the best model information
with open('models/best_model_info.txt', 'w') as f:
    f.write(f"Best model: {best_model_name}\n")
    f.write(f"Accuracy: {best_model_row['accuracy']:.4f}\n")
    f.write(f"Precision: {best_model_row['precision']:.4f}\n")
    f.write(f"Recall: {best_model_row['recall']:.4f}\n")
    f.write(f"F1 Score: {best_model_row['f1']:.4f}\n")

print("\nModel building and evaluation completed successfully!")
