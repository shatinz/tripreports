import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import os

# Configuration
input_file = r'c:\Users\PC\prj\tripreports\old dmt data\report_prototype_similarity.json'
output_dir = r'c:\Users\PC\prj\tripreports\analysis_results'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract prototype_similarity for each report
    rows = []
    for entry in data:
        if 'prototype_similarity' in entry:
            rows.append(entry['prototype_similarity'])
    
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} reports.")
    return df

def calculate_correlations_and_pvalues(df):
    print("Calculating correlations and p-values...")
    numeric_df = df.select_dtypes(include=[np.number])
    columns = numeric_df.columns
    n_cols = len(columns)
    
    corr_matrix = np.zeros((n_cols, n_cols))
    p_value_matrix = np.zeros((n_cols, n_cols))
    
    for i in range(n_cols):
        for j in range(n_cols):
            if i == j:
                corr, p = 1.0, 0.0
            else:
                corr, p = spearmanr(numeric_df.iloc[:, i], numeric_df.iloc[:, j])
            corr_matrix[i, j] = corr
            p_value_matrix[i, j] = p
            
    corr_df = pd.DataFrame(corr_matrix, index=columns, columns=columns)
    p_value_df = pd.DataFrame(p_value_matrix, index=columns, columns=columns)
    
    return corr_df, p_value_df

def bootstrap_correlations(df, n_iterations=1000):
    print(f"Bootstrapping correlations ({n_iterations} iterations)...")
    numeric_df = df.select_dtypes(include=[np.number])
    columns = numeric_df.columns
    n = len(numeric_df)
    
    # Store all bootstrap correlations
    # Shape: (n_iterations, n_pairs) where n_pairs is number of unique pairs
    pairs = []
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            pairs.append((columns[i], columns[j]))
            
    bootstrap_results = {pair: [] for pair in pairs}
    
    for _ in range(n_iterations):
        # Resample with replacement
        sample = numeric_df.sample(n=n, replace=True)
        corr_matrix = sample.corr(method='spearman')
        
        for p1, p2 in pairs:
            bootstrap_results[(p1, p2)].append(corr_matrix.loc[p1, p2])
            
    stats = {}
    for pair, values in bootstrap_results.items():
        lower = np.percentile(values, 2.5)
        upper = np.percentile(values, 97.5)
        mean_corr = np.mean(values)
        stats[pair] = {
            'mean': mean_corr,
            'ci_lower': lower,
            'ci_upper': upper,
            'std': np.std(values)
        }
        
    return stats

def analyze_stability(df, fractions=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0], n_repeats=10):
    print("Analyzing correlation stability...")
    numeric_df = df.select_dtypes(include=[np.number])
    columns = numeric_df.columns
    
    # Track average standard deviation of correlations at each sample size
    stability_stats = []
    
    for frac in fractions:
        sample_std_devs = []
        for _ in range(n_repeats):
            # Sample without replacement
            sample = numeric_df.sample(frac=frac)
            corr = sample.corr(method='spearman')
            # Just take the upper triangle values to represent the state
            vals = corr.values[np.triu_indices_from(corr.values, k=1)]
            sample_std_devs.append(vals)
        
        # Stacking arrays to calculate std dev across repeats for each pair
        stacked = np.stack(sample_std_devs)
        # Calculate std dev of coefficients across the repeats
        std_per_pair = np.std(stacked, axis=0)
        # Average stability (lower std dev = higher stability) across all pairs
        avg_std = np.mean(std_per_pair)
        
        stability_stats.append({
            'fraction': frac,
            'avg_coeff_std': avg_std,
            'n_samples': int(len(df) * frac)
        })
        
    return pd.DataFrame(stability_stats)

def plot_heatmap(corr_df, p_value_df):
    plt.figure(figsize=(16, 14))
    
    # Create labels with significance markers
    annot_labels = corr_df.applymap(lambda x: f"{x:.2f}")
    
    for i in range(len(corr_df)):
        for j in range(len(corr_df)):
            if p_value_df.iloc[i, j] < 0.001:
                annot_labels.iloc[i, j] += '***'
            elif p_value_df.iloc[i, j] < 0.01:
                annot_labels.iloc[i, j] += '**'
            elif p_value_df.iloc[i, j] < 0.05:
                annot_labels.iloc[i, j] += '*'
                
    sns.heatmap(corr_df, annot=annot_labels.values, fmt='', cmap='coolwarm', vmin=-1, vmax=1, square=True)
    plt.title('Spearman Correlation Matrix with Significance (*p<0.05, **p<0.01, ***p<0.001)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap_significance.png'))
    plt.close()
    print("Saved correlation_heatmap_significance.png")

def plot_top_correlations(bootstrap_stats, top_n=15):
    # Sort by absolute correlation strength
    sorted_pairs = sorted(bootstrap_stats.items(), key=lambda x: abs(x[1]['mean']), reverse=True)[:top_n]
    
    labels = [f"{p[0]} vs {p[1]}" for p, _ in sorted_pairs]
    means = [data['mean'] for _, data in sorted_pairs]
    errors = [
        [data['mean'] - data['ci_lower'] for _, data in sorted_pairs],
        [data['ci_upper'] - data['mean'] for _, data in sorted_pairs]
    ]
    
    plt.figure(figsize=(12, 10))
    y_pos = np.arange(len(labels))
    plt.barh(y_pos, means, xerr=errors, align='center', alpha=0.7, capsize=5)
    plt.yticks(y_pos, labels)
    plt.xlabel('Spearman Correlation Coefficient')
    plt.title(f'Top {top_n} Strongest Correlations with 95% Bootstrap CI')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.gca().invert_yaxis()
    plt.savefig(os.path.join(output_dir, 'top_correlations_ci.png'))
    plt.close()
    print("Saved top_correlations_ci.png")

def plot_stability(stability_df):
    plt.figure(figsize=(10, 6))
    plt.plot(stability_df['fraction'], stability_df['avg_coeff_std'], marker='o', linestyle='-')
    plt.xlabel('Fraction of Data Sampled')
    plt.ylabel('Average Std Dev of Correlation Coefficients')
    plt.title('Correlation Stability Analysis')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_stability.png'))
    plt.close()
    print("Saved correlation_stability.png")

def main():
    # 1. Load Data
    df = load_data(input_file)
    
    # 2. Basic Correlation & P-values
    corr_df, p_value_df = calculate_correlations_and_pvalues(df)
    plot_heatmap(corr_df, p_value_df)
    
    # 3. Bootstrap Analysis
    bootstrap_stats = bootstrap_correlations(df)
    plot_top_correlations(bootstrap_stats)
    
    # 4. Stability Analysis
    stability_df = analyze_stability(df)
    plot_stability(stability_df)
    
    print("\nAnalysis Complete!")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
