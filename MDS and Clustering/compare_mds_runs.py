import os
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt

# This script loads MDS metrics from specific runs, analyzes them, and generates plots for R_squared, Stress, Inertia, and Silhouette_Score across different dimensions.

def load_metrics(base_dir):
    """Load mds_metrics.csv files specifically from run_19, run_20, and run_21 directories."""
    runs_to_include = ['1_run_23','2_run_19', '3_run_20', '4_run_24', '5_run_22', '6_run_26'] # in ascenfding order: 1 - 6 dimensions
    all_metrics = {}

    for run in runs_to_include:
        metrics_file = os.path.join(base_dir, run, 'mds_metrics.csv')
        if os.path.exists(metrics_file):
            print(f"Loading metrics from {metrics_file}")
            metrics_df = pd.read_csv(metrics_file)
            all_metrics[run] = metrics_df
        else:
            print(f"Metrics file not found for {run}")

    # Combine all metrics into a single DataFrame
    combined_metrics = pd.concat(all_metrics, names=['Run', 'Index'])
    combined_metrics.reset_index(level=1, drop=True, inplace=True)
    return combined_metrics

def analyze_metrics(combined_metrics):
    """Analyze the metrics and find the best run."""
    # Exclude non-numeric columns before grouping
    numeric_metrics = combined_metrics.select_dtypes(include=[np.number])
    average_metrics = numeric_metrics.groupby(combined_metrics.index).mean()

    # Find the best run
    best_run = average_metrics.iloc[0]
    print("\nBest Run:")
    print(best_run)

    return average_metrics, best_run

def plot_metrics(average_metrics, output_dir):
    """Plot R_squared, Stress, Inertia, and Silhouette_Score across runs in separate diagrams and save the plots."""
    # Extract dimensions from the run names (e.g., '3_run_20' -> 3)
    dimensions = [int(run.split('_')[0]) for run in average_metrics.index]
    
    # Plot R_squared
    plt.figure(figsize=(10, 6))
    plt.plot(dimensions, average_metrics['R_squared'], marker='o', label='R_squared', color='blue')
    plt.xlabel('Dimensions after MDS', fontsize=14)
    plt.ylabel('R_squared', fontsize=14)
    plt.xticks(dimensions)  # Ensure x-axis numbers have no decimal
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'r_squared_across_runs.png'), dpi=300)
    plt.close()
    print(f"R_squared plot saved to {output_dir}/r_squared_across_runs.png")
    
    # Plot Stress
    plt.figure(figsize=(10, 6))
    plt.plot(dimensions, average_metrics['Stress'], marker='o', label='Stress', color='orange')
    plt.xlabel('Dimensions after MDS', fontsize=14)
    plt.ylabel('Stress', fontsize=14)
    plt.xticks(dimensions)  # Ensure x-axis numbers have no decimal
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'stress_across_runs.png'), dpi=300)
    plt.close()
    print(f"Stress plot saved to {output_dir}/stress_across_runs.png")
    
    # Plot Inertia
    plt.figure(figsize=(10, 6))
    plt.plot(dimensions, average_metrics['Inertia'], marker='o', label='Inertia', color='green')
    plt.xlabel('Dimensions after MDS', fontsize=14)
    plt.ylabel('Inertia', fontsize=14)
    plt.xticks(dimensions)  # Ensure x-axis numbers have no decimal
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'inertia_across_runs.png'), dpi=300)
    plt.close()
    print(f"Inertia plot saved to {output_dir}/inertia_across_runs.png")
    
    # Plot Silhouette_Score
    plt.figure(figsize=(10, 6))
    plt.plot(dimensions, average_metrics['Silhouette_Score'], marker='o', label='Silhouette_Score', color='red')
    plt.xlabel('Dimensions after MDS', fontsize=14)
    plt.ylabel('Silhouette Score', fontsize=14)
    plt.xticks(dimensions)  # Ensure x-axis numbers have no decimal
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'silhouette_score_across_runs.png'), dpi=300)
    plt.close()
    print(f"Silhouette_Score plot saved to {output_dir}/silhouette_score_across_runs.png")

def main():
    base_dir = '/home/hmueller2/ibc_code/ibc_output_MDS'
    output_dir = '/home/hmueller2/ibc_code/images_presentation'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metrics from all runs
    combined_metrics = load_metrics(base_dir)
    
    # Analyze metrics and find the best run
    average_metrics, best_run = analyze_metrics(combined_metrics)
    
    # Print the average metrics
    print("\nAverage Metrics Across Runs:")
    print(average_metrics)
    
    # Plot the metrics and save the plots
    plot_metrics(average_metrics, output_dir)

if __name__ == "__main__":
    main()