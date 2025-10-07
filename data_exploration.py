import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set the style and color palette
sns.set_style("whitegrid")
pink_blue_palette = ["#ADD8E6", "#FFB6C1"]  # Light blue for male (1), Light pink for female (2)

# Create output directory for visualizations
os.makedirs('visualizations', exist_ok=True)

# Load the dataset
df = pd.read_csv(r"C:\Users\bochra.chemam\Downloads\Gender_Prediction_Project\nat2022.csv", sep=';')

# Display basic information about the dataset
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nSummary statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Convert 'annais' to numeric if it's a year
if df['annais'].dtype == 'object':
    try:
        df['annais'] = pd.to_numeric(df['annais'])
        print("\nConverted 'annais' to numeric type")
    except:
        print("\nCouldn't convert 'annais' to numeric type")

# Univariate Analysis

# 1. Gender distribution (sexe)
plt.figure(figsize=(10, 6))
gender_counts = df['sexe'].value_counts()
ax = sns.barplot(x=gender_counts.index, y=gender_counts.values, palette=pink_blue_palette)
plt.title('Gender Distribution', fontsize=15)
plt.xlabel('Gender (1: Male, 2: Female)', fontsize=12)
plt.ylabel('Count', fontsize=12)
for i, v in enumerate(gender_counts.values):
    ax.text(i, v + 0.1, f'{v:,}', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('visualizations/gender_distribution.png')
plt.close()

# 2. Year distribution (annais)
plt.figure(figsize=(14, 7))
year_counts = df['annais'].value_counts().sort_index()
sns.lineplot(x=year_counts.index, y=year_counts.values, color='purple')
plt.title('Distribution of Records by Year', fontsize=15)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Records', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig('visualizations/year_distribution.png')
plt.close()

# 3. Top 20 most common names
plt.figure(figsize=(14, 8))
name_counts = df['preusuel'].value_counts().head(20)
ax = sns.barplot(x=name_counts.values, y=name_counts.index, palette='viridis')
plt.title('Top 20 Most Common Names', fontsize=15)
plt.xlabel('Count', fontsize=12)
plt.ylabel('Name', fontsize=12)
for i, v in enumerate(name_counts.values):
    ax.text(v + 0.1, i, f'{v:,}', va='center', fontsize=10)
plt.tight_layout()
plt.savefig('visualizations/top_20_names.png')
plt.close()

# 4. Distribution of 'nombre' (count)
plt.figure(figsize=(12, 6))
sns.histplot(df['nombre'], bins=50, kde=True, color=pink_blue_palette[0])
plt.title('Distribution of Name Counts', fontsize=15)
plt.xlabel('Count', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig('visualizations/nombre_distribution.png')
plt.close()

# 5. Distribution of name lengths
df['name_length'] = df['preusuel'].str.len()
plt.figure(figsize=(12, 6))
sns.histplot(df['name_length'].dropna(), bins=30, kde=True, color=pink_blue_palette[1])
plt.title('Distribution of Name Lengths', fontsize=15)
plt.xlabel('Name Length', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig('visualizations/name_length_distribution.png')
plt.close()

# Bivariate Analysis

# 1. Gender distribution over years
plt.figure(figsize=(14, 7))
gender_year = df.groupby(['annais', 'sexe'])['nombre'].sum().unstack()
gender_year.columns = ['Male', 'Female']
gender_year.plot(kind='line', color=pink_blue_palette)
plt.title('Gender Distribution Over Years', fontsize=15)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(True)
plt.legend(title='Gender')
plt.tight_layout()
plt.savefig('visualizations/gender_over_years.png')
plt.close()

# 2. Average name length by gender
plt.figure(figsize=(10, 6))
avg_length_by_gender = df.groupby('sexe')['name_length'].mean()
ax = sns.barplot(x=avg_length_by_gender.index, y=avg_length_by_gender.values, palette=pink_blue_palette)
plt.title('Average Name Length by Gender', fontsize=15)
plt.xlabel('Gender (1: Male, 2: Female)', fontsize=12)
plt.ylabel('Average Name Length', fontsize=12)
for i, v in enumerate(avg_length_by_gender.values):
    ax.text(i, v + 0.1, f'{v:.2f}', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('visualizations/avg_name_length_by_gender.png')
plt.close()

# 3. Name length distribution by gender
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='name_length', hue='sexe', palette=pink_blue_palette, kde=True, element='step')
plt.title('Name Length Distribution by Gender', fontsize=15)
plt.xlabel('Name Length', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend(title='Gender', labels=['Male', 'Female'])
plt.grid(True)
plt.tight_layout()
plt.savefig('visualizations/name_length_dist_by_gender.png')
plt.close()

# 4. Top 10 names by gender
plt.figure(figsize=(14, 10))
top_male_names = df[df['sexe'] == 1].groupby('preusuel')['nombre'].sum().sort_values(ascending=False).head(10)
top_female_names = df[df['sexe'] == 2].groupby('preusuel')['nombre'].sum().sort_values(ascending=False).head(10)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Male names
sns.barplot(x=top_male_names.values, y=top_male_names.index, color=pink_blue_palette[0], ax=ax1)
ax1.set_title('Top 10 Male Names', fontsize=15)
ax1.set_xlabel('Count', fontsize=12)
ax1.set_ylabel('Name', fontsize=12)
for i, v in enumerate(top_male_names.values):
    ax1.text(v + 0.1, i, f'{v:,}', va='center', fontsize=10)

# Female names
sns.barplot(x=top_female_names.values, y=top_female_names.index, color=pink_blue_palette[1], ax=ax2)
ax2.set_title('Top 10 Female Names', fontsize=15)
ax2.set_xlabel('Count', fontsize=12)
ax2.set_ylabel('', fontsize=12)
for i, v in enumerate(top_female_names.values):
    ax2.text(v + 0.1, i, f'{v:,}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('visualizations/top_names_by_gender.png')
plt.close()

# 5. Trend of specific popular names over time
top_names = df.groupby('preusuel')['nombre'].sum().sort_values(ascending=False).head(5).index
plt.figure(figsize=(14, 7))
for name in top_names:
    name_data = df[df['preusuel'] == name].groupby('annais')['nombre'].sum()
    plt.plot(name_data.index, name_data.values, label=name)
plt.title('Trend of Top 5 Popular Names Over Time', fontsize=15)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Name')
plt.grid(True)
plt.tight_layout()
plt.savefig('visualizations/top_names_trend.png')
plt.close()

# 6. First letter distribution by gender
df['first_letter'] = df['preusuel'].str[0].str.upper()
first_letter_gender = df.groupby(['first_letter', 'sexe'])['nombre'].sum().unstack().fillna(0)
first_letter_gender.columns = ['Male', 'Female']
first_letter_gender = first_letter_gender.sort_index()

plt.figure(figsize=(16, 8))
first_letter_gender.plot(kind='bar', color=pink_blue_palette)
plt.title('First Letter Distribution by Gender', fontsize=15)
plt.xlabel('First Letter', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Gender')
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig('visualizations/first_letter_by_gender.png')
plt.close()

# 7. Last letter distribution by gender
df['last_letter'] = df['preusuel'].str[-1].str.upper()
last_letter_gender = df.groupby(['last_letter', 'sexe'])['nombre'].sum().unstack().fillna(0)
last_letter_gender.columns = ['Male', 'Female']
last_letter_gender = last_letter_gender.sort_index()

plt.figure(figsize=(16, 8))
last_letter_gender.plot(kind='bar', color=pink_blue_palette)
plt.title('Last Letter Distribution by Gender', fontsize=15)
plt.xlabel('Last Letter', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Gender')
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig('visualizations/last_letter_by_gender.png')
plt.close()

print("\nData exploration and visualization completed. Check the 'visualizations' directory for the output.")
