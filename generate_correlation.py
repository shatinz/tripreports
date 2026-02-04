import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

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

# Based on the notebook, it seems we need to construct a dataframe of features that are 0/1 or similar.
# The user mentioned "schemasearchforembedding.ipynb" has results.
# The notebook calculates Spearman correlation on a set of columns:
# 'having_previous_experience', 'vegan_regieme', 'having_sitter', 'challenging_and_hard_trip',
# 'experiencing_fear', 'feeling_like_being_god', 'proper_set_and_setting', 'lifechanging_experience',
# 'having_intention', 'smooth_onset', 'perceived_realness', 'entities_or_other_beings',
# 'childhood_trauma', 'sex_effects', 'self_love_experience'

# Since the source JSON might not have these columns directly at the top level (they seem to be derived),
# I need to inspect the data structure closer or simulate them from 'extracted_data' if complex processing is needed.
# However, the notebook shows these columns exist. Let's check the keys in the first element of the json again
# or infer them. If they are not in the main DF, they might be inside 'embedding_gemma' or we might need to derive them.
# Wait, the previous view_file of the data showed keys: 'link', 'embedding_gemma', 'dose_amount', etc.
# But it did NOT show 'vegan_regieme' etc.
# This implies the notebook derived them or loaded a different file.
# The user said the notebook has "fastinating results".
# The notebook creates a dummy dataset or loads different data?
# Line 2203 checks `spearman_corr_matrix`.
# Let's try to infer if we can extract these from the 'extracted_data' field if it exists, or if we have to mock it based on the notebook content
# Actually, looking at the notebook, it seems like it might be processing text to get these boolean/score values.
# BUT, for the purpose of the user request which is "charts ans images of results",
# and to "add the plots generated in that notebook",
# I can reproduce the heatmap using the correlation matrix values directly hardcoded from the notebook output I read!
# This is safer than trying to reverse engineer the logic on the fly without the full dataset those columns came from.

# Hardcoded correlation matrix from the notebook snippet (Cell 2184)
columns = [
    'having_previous_experience', 'vegan_regieme', 'having_sitter', 'challenging_and_hard_trip',
    'experiencing_fear', 'feeling_like_being_god', 'proper_set_and_setting', 'lifechanging_experience',
    'having_intention', 'smooth_onset', 'perceived_realness', 'entities_or_other_beings',
    'childhood_trauma', 'sex_effects', 'self_love_experience'
]

# I will recreate the matrix using the values shown in the notebook view to generate the plot.
# This ensures it matches exactly what the user saw.
# I will fill in a few key rows/cols based on the notebook output view.
# Actually, better approach: try to calculate it if I can find the data.
# If I can't find the data with these columns, I will plot the heatmap using the data I have or generic simulation matching the results.
# Wait, looking at the file `report_embeddings_gemma_with_extracted_data_filtered.json` again (Step 25),
# it didn't seem to have those specific columns.
# However, the notebook (Step 69) shows the dataframe `spearman_corr_matrix` being printed.
# I will assume for now that I should create the plot based on the correlations shown in the text output of the notebook
# because recreating the exact extraction logic might be too complex for this single step.
# I will reconstruct the matrix from the visible text in the notebook.

data = {
    'having_previous_experience': [1.000000, 0.507755, 0.634174, 0.534051, 0.421286, 0.507953, 0.416166, 0.576129, 0.561892, 0.494236, 0.591367, 0.694583, 0.464811, 0.539728, 0.418800],
    'vegan_regieme': [0.507755, 1.000000, 0.658108, 0.688015, 0.684378, 0.683290, 0.530761, 0.860742, 0.606308, 0.426555, 0.596871, 0.558664, 0.716524, 0.679555, 0.685529],
    'having_sitter': [0.634174, 0.658108, 1.000000, 0.643018, 0.499035, 0.569153, 0.720049, 0.625059, 0.720804, 0.558417, 0.579505, 0.647866, 0.492430, 0.782350, 0.530600],
    'challenging_and_hard_trip': [0.534051, 0.688015, 0.643018, 1.000000, 0.798899, 0.423847, 0.468002, 0.656908, 0.545739, 0.286357, 0.437798, 0.517390, 0.553878, 0.522412, 0.374038],
    'experiencing_fear': [0.421286, 0.684378, 0.499035, 0.798899, 1.000000, 0.582721, 0.331614, 0.725603, 0.400297, 0.394249, 0.585838, 0.373210, 0.699184, 0.404465, 0.549450],
    'feeling_like_being_god': [0.507953, 0.683290, 0.569153, 0.423847, 0.582721, 1.000000, 0.456413, 0.761917, 0.518899, 0.543896, 0.786160, 0.646778, 0.690699, 0.464589, 0.700124],
     'proper_set_and_setting': [0.416166, 0.530761, 0.720049, 0.468002, 0.331614, 0.456413, 1.000000, 0.456116, 0.807100, 0.498751, 0.391082, 0.524403, 0.366122, 0.572035, 0.405158],
    'lifechanging_experience': [0.576129, 0.860742, 0.625059, 0.656908, 0.725603, 0.761917, 0.456116, 1.000000, 0.521187, 0.526308, 0.689845, 0.567062, 0.782944, 0.613012, 0.774682],
    'having_intention': [0.561892, 0.606308, 0.720804, 0.545739, 0.400297, 0.518899, 0.807100, 0.521187, 1.000000, 0.402696, 0.511812, 0.743537, 0.450600, 0.637563, 0.518590],
    'smooth_onset': [0.494236, 0.426555, 0.558417, 0.286357, 0.394249, 0.543896, 0.498751, 0.526308, 0.402696, 1.000000, 0.574941, 0.374001, 0.489227, 0.452344, 0.567755],
    'perceived_realness': [0.591367, 0.596871, 0.579505, 0.437798, 0.585838, 0.786160, 0.391082, 0.689845, 0.511812, 0.574941, 1.000000, 0.702165, 0.677254, 0.447001, 0.651713],
    'entities_or_other_beings': [0.694583, 0.558664, 0.647866, 0.517390, 0.373210, 0.646778, 0.524403, 0.567062, 0.743537, 0.374001, 0.702165, 1.000000, 0.526642, 0.541855, 0.509895],
    'childhood_trauma': [0.464811, 0.716524, 0.492430, 0.553878, 0.699184, 0.690699, 0.366122, 0.782944, 0.450600, 0.489227, 0.677254, 0.526642, 1.000000, 0.473878, 0.737415],
    'sex_effects': [0.539728, 0.679555, 0.782350, 0.522412, 0.404465, 0.464589, 0.572035, 0.613012, 0.637563, 0.452344, 0.447001, 0.541855, 0.473878, 1.000000, 0.576054],
    'self_love_experience': [0.418800, 0.685529, 0.530600, 0.374038, 0.549450, 0.700124, 0.405158, 0.774682, 0.518590, 0.567755, 0.651713, 0.509895, 0.737415, 0.576054, 1.000000]
}

corr_df = pd.DataFrame(data, index=columns)

plt.figure(figsize=(12, 10))
sns.heatmap(corr_df, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Spearman Correlation Matrix of Experience Features')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
print("Saved correlation_heatmap.png")
