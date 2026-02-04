import json
import pandas as pd
import matplotlib.pyplot as plt
import os

# Define paths
data_path = 'dmt data/report_embeddings_gemma_with_extracted_data_filtered.json'
output_dir = 'images'

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load data
print(f"Loading data from {data_path}...")
with open(data_path, 'r') as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)

# Print columns to verify
print("Columns found:", df.columns)

# 1. Substance Used Distribution (Top 10)
plt.figure(figsize=(12, 6))
substance_counts = df['substance_used'].value_counts().head(10)
substance_counts.plot(kind='bar', color='skyblue')
plt.title('Top 10 Substances Used in Reports')
plt.xlabel('Substance')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'substance_distribution.png'))
print("Saved substance_distribution.png")
plt.close()

# 2. Age Distribution
# Clean age column (convert to numeric, coerce errors)
df['age'] = pd.to_numeric(df['age'], errors='coerce')
plt.figure(figsize=(10, 6))
df['age'].plot(kind='hist', bins=20, color='lightgreen', edgecolor='black')
plt.title('Age Distribution of Reporters')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'age_distribution.png'))
print("Saved age_distribution.png")
plt.close()

# 3. Gender Distribution
plt.figure(figsize=(8, 6))
gender_counts = df['gender'].value_counts()
gender_counts.plot(kind='bar', color='salmon')
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'gender_distribution.png'))
print("Saved gender_distribution.png")
plt.close()

# 4. Year of Experience
# Clean year column
df['year_of_experience'] = pd.to_numeric(df['year_of_experience'], errors='coerce')
# Filter obvious outliers (e.g. year < 1960 or > 2025) if necessary, but histogram handles range.
plt.figure(figsize=(12, 6))
df['year_of_experience'].plot(kind='hist', bins=30, color='orange', edgecolor='black')
plt.title('Year of Experience Distribution')
plt.xlabel('Year')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'year_distribution.png'))
print("Saved year_distribution.png")
plt.close()

print("All charts generated successfully.")
