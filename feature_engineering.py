import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import os

# Set the style and color palette - CORRECTED: blue for male, pink for female
sns.set_style("whitegrid")
gender_palette = ["#ADD8E6", "#FFB6C1"]  # Blue for male (1), Pink for female (2)

# Create output directory for feature engineering
os.makedirs('features', exist_ok=True)

# Load the dataset
print("Loading dataset...")
df = pd.read_csv(r"C:\Users\bochra.chemam\Downloads\Gender_Prediction_Project\nat2022.csv", sep=';')

# Check for missing values and handle them
print("\nChecking for missing values:")
print(df.isnull().sum())

# Drop rows with missing values in 'preusuel' column
df = df.dropna(subset=['preusuel'])
print(f"Dataset shape after dropping missing values: {df.shape}")

# Check class distribution
print("\nClass distribution (gender):")
gender_counts = df['sexe'].value_counts()
print(gender_counts)
print(f"Male percentage: {gender_counts[1]/len(df)*100:.2f}%")
print(f"Female percentage: {gender_counts[2]/len(df)*100:.2f}%")

# Visualize the corrected class distribution
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=gender_counts.index, y=gender_counts.values, palette=gender_palette)
plt.title('Gender Distribution (Corrected Colors)', fontsize=15)
plt.xlabel('Gender (1: Male, 2: Female)', fontsize=12)
plt.ylabel('Count', fontsize=12)
for i, v in enumerate(gender_counts.values):
    ax.text(i, v + 0.1, f'{v:,}', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('features/corrected_gender_distribution.png')
plt.close()

# Calculate imbalance ratio
imbalance_ratio = gender_counts.max() / gender_counts.min()
print(f"\nImbalance ratio: {imbalance_ratio:.2f}")

# Feature Engineering

# 1. Basic name features
print("\nExtracting basic name features...")
df['name_length'] = df['preusuel'].str.len()
df['first_letter'] = df['preusuel'].str[0].str.lower()
df['last_letter'] = df['preusuel'].str[-1].str.lower()
df['vowels_count'] = df['preusuel'].str.lower().str.count(r'[aeiouàáâäæãåāèéêëēėęîïíīįìôöòóœøōõûüùúūÿ]')
df['consonants_count'] = df['preusuel'].str.lower().str.count(r'[bcdfghjklmnpqrstvwxzç]')
df['vowel_ratio'] = df['vowels_count'] / df['name_length']

# 2. Suffix features (last 2 and 3 letters)
df['suffix_2'] = df['preusuel'].str[-2:].str.lower()
df['suffix_3'] = df['preusuel'].str[-3:].str.lower()

# 3. Prefix features (first 2 and 3 letters)
df['prefix_2'] = df['preusuel'].str[:2].str.lower()
df['prefix_3'] = df['preusuel'].str[:3].str.lower()

# 4. Contains specific patterns
df['contains_double_letter'] = df['preusuel'].str.lower().apply(
    lambda x: any(i == j for i, j in zip(x, x[1:]))
)
df['ends_with_vowel'] = df['last_letter'].str.lower().str.contains(r'[aeiouàáâäæãåāèéêëēėęîïíīįìôöòóœøōõûüùúūÿ]')
df['starts_with_vowel'] = df['first_letter'].str.lower().str.contains(r'[aeiouàáâäæãåāèéêëēėęîïíīįìôöòóœøōõûüùúūÿ]')

# 5. Special French name features
df['ends_with_ine'] = df['preusuel'].str.lower().str.endswith('ine')
df['ends_with_a'] = df['preusuel'].str.lower().str.endswith('a')
df['ends_with_e'] = df['preusuel'].str.lower().str.endswith('e')
df['ends_with_ie'] = df['preusuel'].str.lower().str.endswith('ie')
df['ends_with_ette'] = df['preusuel'].str.lower().str.endswith('ette')
df['ends_with_elle'] = df['preusuel'].str.lower().str.endswith('elle')
df['ends_with_eur'] = df['preusuel'].str.lower().str.endswith('eur')
df['ends_with_on'] = df['preusuel'].str.lower().str.endswith('on')
df['ends_with_nt'] = df['preusuel'].str.lower().str.endswith('nt')

# Visualize some of the engineered features
print("\nVisualizing engineered features...")

# 1. Name length distribution by gender (corrected colors)
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='name_length', hue='sexe', palette=gender_palette, kde=True, element='step')
plt.title('Name Length Distribution by Gender (Corrected Colors)', fontsize=15)
plt.xlabel('Name Length', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend(title='Gender', labels=['Male', 'Female'])
plt.grid(True)
plt.tight_layout()
plt.savefig('features/name_length_by_gender_corrected.png')
plt.close()

# 2. Vowel ratio by gender
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='sexe', y='vowel_ratio', palette=gender_palette)
plt.title('Vowel Ratio by Gender', fontsize=15)
plt.xlabel('Gender (1: Male, 2: Female)', fontsize=12)
plt.ylabel('Vowel Ratio', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig('features/vowel_ratio_by_gender.png')
plt.close()

# 3. Ending with vowel by gender
plt.figure(figsize=(10, 6))
ends_vowel_by_gender = pd.crosstab(df['sexe'], df['ends_with_vowel'])
ends_vowel_by_gender_pct = ends_vowel_by_gender.div(ends_vowel_by_gender.sum(axis=1), axis=0) * 100
ends_vowel_by_gender_pct.plot(kind='bar', stacked=True, color=['gray', 'green'])
plt.title('Names Ending with Vowel by Gender', fontsize=15)
plt.xlabel('Gender (1: Male, 2: Female)', fontsize=12)
plt.ylabel('Percentage', fontsize=12)
plt.legend(title='Ends with Vowel', labels=['No', 'Yes'])
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig('features/ends_with_vowel_by_gender.png')
plt.close()

# 4. Top suffixes by gender
suffix_3_by_gender = df.groupby(['suffix_3', 'sexe'])['nombre'].sum().unstack().fillna(0)
suffix_3_by_gender.columns = ['Male', 'Female']
top_suffixes = suffix_3_by_gender.sum(axis=1).sort_values(ascending=False).head(15).index

plt.figure(figsize=(14, 8))
suffix_3_by_gender.loc[top_suffixes].plot(kind='bar', color=gender_palette)
plt.title('Top 15 3-Letter Suffixes by Gender', fontsize=15)
plt.xlabel('Suffix', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Gender')
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig('features/top_suffixes_by_gender.png')
plt.close()

# 5. Special French endings by gender
french_endings = ['ends_with_ine', 'ends_with_a', 'ends_with_e', 
                 'ends_with_ie', 'ends_with_ette', 'ends_with_elle',
                 'ends_with_eur', 'ends_with_on', 'ends_with_nt']

french_endings_data = []
for ending in french_endings:
    ending_by_gender = pd.crosstab(df['sexe'], df[ending])
    ending_by_gender_pct = ending_by_gender.div(ending_by_gender.sum(axis=1), axis=0) * 100
    french_endings_data.append({
        'ending': ending.replace('ends_with_', ''),
        'male_pct': ending_by_gender_pct.loc[1, True],
        'female_pct': ending_by_gender_pct.loc[2, True]
    })

french_endings_df = pd.DataFrame(french_endings_data)
french_endings_df = french_endings_df.sort_values('female_pct', ascending=False)

plt.figure(figsize=(14, 8))
x = np.arange(len(french_endings_df))
width = 0.35
fig, ax = plt.subplots(figsize=(14, 8))
ax.bar(x - width/2, french_endings_df['male_pct'], width, label='Male', color=gender_palette[0])
ax.bar(x + width/2, french_endings_df['female_pct'], width, label='Female', color=gender_palette[1])
ax.set_xticks(x)
ax.set_xticklabels(french_endings_df['ending'])
ax.set_ylabel('Percentage of Names')
ax.set_title('Percentage of Names with Special French Endings by Gender')
ax.legend()
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig('features/french_endings_by_gender.png')
plt.close()

# Prepare data for modeling
print("\nPreparing data for modeling...")

# Create a new dataframe with only the necessary columns
model_df = df[['sexe', 'preusuel', 'nombre']].copy()

# Expand the dataset based on 'nombre' (count)
# This is important to properly represent the actual distribution of names
print("Expanding dataset based on name counts...")
# To avoid memory issues, we'll sample a portion of the data
# First, calculate total counts
total_count = model_df['nombre'].sum()
print(f"Total count in original dataset: {total_count:,}")

# Set a reasonable sample size (e.g., 1 million records)
sample_size = 500000  # Reduced sample size to manage memory better

# Create a weighted sampling approach that doesn't try to sample more than available
print("Creating weighted sample...")
# Create a probability distribution based on nombre
model_df['sampling_weight'] = model_df['nombre'] / model_df['nombre'].sum()

# Sample with replacement to ensure we get the desired sample size
# This is appropriate since we're trying to represent the full distribution
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
plt.savefig('features/expanded_gender_distribution.png')
plt.close()

# Feature extraction for character-level analysis
print("\nExtracting character-level features...")

# 1. Character n-grams using CountVectorizer
# This will be useful for both traditional ML and as input to LSTM
print("Generating character n-grams...")
char_vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 3))
X_char_ngrams = char_vectorizer.fit_transform(expanded_df['preusuel'])
char_feature_names = char_vectorizer.get_feature_names_out()
print(f"Number of character n-gram features: {len(char_feature_names)}")

# Save the vectorizer for later use in deployment
import pickle
with open('features/char_vectorizer.pkl', 'wb') as f:
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
with open('features/char_mappings.pkl', 'wb') as f:
    pickle.dump({'char_to_idx': char_to_idx, 'idx_to_char': idx_to_char}, f)

# 3. Manual feature extraction for traditional ML models
print("Extracting manual features...")
# Function to extract features from a name
def extract_features(name):
    name = name.lower()
    features = {
        'length': len(name),
        'vowels_count': sum(1 for c in name if c in 'aeiouàáâäæãåāèéêëēėęîïíīįìôöòóœøōõûüùúūÿ'),
        'consonants_count': sum(1 for c in name if c in 'bcdfghjklmnpqrstvwxzç'),
        'starts_with_vowel': int(name[0] in 'aeiouàáâäæãåāèéêëēėęîïíīįìôöòóœøōõûüùúūÿ'),
        'ends_with_vowel': int(name[-1] in 'aeiouàáâäæãåāèéêëēėęîïíīįìôöòóœøōõûüùúūÿ'),
        'contains_double_letter': int(any(i == j for i, j in zip(name, name[1:]))),
    }
    
    # Add first and last letter features
    for c in sorted(all_chars):
        features[f'first_{c}'] = int(name[0] == c)
        features[f'last_{c}'] = int(name[-1] == c)
    
    # Add suffix and prefix features
    if len(name) >= 2:
        features[f'suffix_2_{name[-2:]}'] = 1
        features[f'prefix_2_{name[:2]}'] = 1
    if len(name) >= 3:
        features[f'suffix_3_{name[-3:]}'] = 1
        features[f'prefix_3_{name[:3]}'] = 1
    
    # Special French endings
    features['ends_with_ine'] = int(name.endswith('ine'))
    features['ends_with_a'] = int(name.endswith('a'))
    features['ends_with_e'] = int(name.endswith('e'))
    features['ends_with_ie'] = int(name.endswith('ie'))
    features['ends_with_ette'] = int(name.endswith('ette'))
    features['ends_with_elle'] = int(name.endswith('elle'))
    features['ends_with_eur'] = int(name.endswith('eur'))
    features['ends_with_on'] = int(name.endswith('on'))
    features['ends_with_nt'] = int(name.endswith('nt'))
    
    return features

# Apply feature extraction to a sample for demonstration
sample_size = min(10000, len(expanded_df))
sample_df = expanded_df.sample(sample_size, random_state=42)
manual_features = []
for name in sample_df['preusuel']:
    manual_features.append(extract_features(name))

manual_features_df = pd.DataFrame(manual_features)
print(f"Manual features dataframe shape: {manual_features_df.shape}")

# Save a sample of the manual features for inspection
manual_features_df.head(20).to_csv('features/manual_features_sample.csv', index=False)

# Prepare train/test split for modeling
print("\nPreparing train/test split...")
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
if imbalance_ratio > 1.5:
    print("\nHandling class imbalance...")
    # Option 1: Class weights (will be used in model training)
    # Calculate class weights
    class_weights = {0: len(y_train) / (2 * (len(y_train) - sum(y_train))),
                     1: len(y_train) / (2 * sum(y_train))}
    print(f"Class weights: {class_weights}")
    
    # Option 2: SMOTE (Synthetic Minority Over-sampling Technique)
    print("Applying SMOTE to balance classes...")
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
    with open('features/train_test_data.pkl', 'wb') as f:
        pickle.dump({
            'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test,
            'X_train_smote': X_train_smote, 'y_train_smote': y_train_smote,
            'class_weights': class_weights
        }, f)
else:
    # Save original datasets
    with open('features/train_test_data.pkl', 'wb') as f:
        pickle.dump({
            'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test
        }, f)

# Prepare data for LSTM model
print("\nPreparing data for LSTM model...")
# Find maximum name length
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

# Convert a sample of names to sequences for demonstration
sample_size = min(10000, len(expanded_df))
sample_df = expanded_df.sample(sample_size, random_state=42)
X_lstm_sample = np.array([name_to_sequence(name, max_name_length) for name in sample_df['preusuel']])
y_lstm_sample = (sample_df['sexe'].values == 1).astype(int)

print(f"LSTM input sample shape: {X_lstm_sample.shape}")

# Save LSTM sample data
with open('features/lstm_sample_data.pkl', 'wb') as f:
    pickle.dump({
        'X_lstm_sample': X_lstm_sample,
        'y_lstm_sample': y_lstm_sample,
        'max_name_length': max_name_length
    }, f)

print("\nFeature engineering completed successfully!")
