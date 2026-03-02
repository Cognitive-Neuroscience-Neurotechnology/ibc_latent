"""
Visualization utilities for Multiple Demand system maps.

This script provides functions to visualize and analyze MD maps.
"""

import os
import sys
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import argparse


def get_percentile_threshold(data, percentile=95):
    """Calculate threshold at given percentile of positive values."""
    positive_data = data[data > 0]
    if len(positive_data) == 0:
        return 0
    return np.percentile(positive_data, percentile)


def summarize_md_map(md_map_path):
    """Print summary statistics for an MD map."""
    img = nib.load(md_map_path)
    data = img.get_fdata()
    
    if data.shape[0] == 1:
        data = data[0]
    
    print(f"\nSummary for: {os.path.basename(md_map_path)}")
    print("=" * 60)
    print(f"Shape: {data.shape}")
    print(f"Mean: {np.mean(data):.4f}")
    print(f"Std: {np.std(data):.4f}")
    print(f"Min: {np.min(data):.4f}")
    print(f"Max: {np.max(data):.4f}")
    print(f"Median: {np.median(data):.4f}")
    
    # Percentiles
    positive_data = data[data > 0]
    if len(positive_data) > 0:
        print(f"\nPositive values only:")
        print(f"  Mean: {np.mean(positive_data):.4f}")
        print(f"  95th percentile: {np.percentile(positive_data, 95):.4f}")
        print(f"  99th percentile: {np.percentile(positive_data, 99):.4f}")
        print(f"  Count: {len(positive_data)} / {len(data)} ({100*len(positive_data)/len(data):.1f}%)")


def plot_md_histogram(md_map_path, output_path=None, threshold=None):
    """Plot histogram of z-scores in MD map."""
    img = nib.load(md_map_path)
    data = img.get_fdata()
    
    if data.shape[0] == 1:
        data = data[0]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Full distribution
    axes[0].hist(data, bins=100, alpha=0.7, color='blue', edgecolor='black')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
    if threshold:
        axes[0].axvline(threshold, color='green', linestyle='--', linewidth=2, 
                       label=f'Threshold: {threshold:.2f}')
    axes[0].set_xlabel('Z-score')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Full Distribution')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Positive values only
    positive_data = data[data > 0]
    if len(positive_data) > 0:
        axes[1].hist(positive_data, bins=100, alpha=0.7, color='green', edgecolor='black')
        if threshold:
            axes[1].axvline(threshold, color='red', linestyle='--', linewidth=2,
                           label=f'Threshold: {threshold:.2f}')
        axes[1].set_xlabel('Z-score')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Positive Values Only')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    
    plt.suptitle(f'MD Map: {os.path.basename(md_map_path)}')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved histogram to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def compare_subjects(md_maps_dir, output_dir=None):
    """Compare MD maps across subjects."""
    # Find all subject MD maps
    subject_dirs = sorted([d for d in os.listdir(md_maps_dir) 
                          if d.startswith('sub-') and os.path.isdir(os.path.join(md_maps_dir, d))])
    
    if not subject_dirs:
        print("No subject directories found")
        return
    
    subjects = []
    means = []
    maxs = []
    n_contrasts = []
    
    for subj_dir in subject_dirs:
        subject = subj_dir.replace('sub-', '')
        md_map_path = os.path.join(md_maps_dir, subj_dir, f'sub-{subject}_MD_mean.dscalar.nii')
        info_path = os.path.join(md_maps_dir, subj_dir, f'sub-{subject}_MD_contrasts.txt')
        
        if not os.path.exists(md_map_path):
            continue
        
        # Load MD map
        img = nib.load(md_map_path)
        data = img.get_fdata()
        if data.shape[0] == 1:
            data = data[0]
        
        subjects.append(subject)
        means.append(np.mean(data[data > 0]))
        maxs.append(np.max(data))
        
        # Count contrasts
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if 'Number of contrasts:' in line:
                        n = int(line.split(':')[1].strip())
                        n_contrasts.append(n)
                        break
        else:
            n_contrasts.append(0)
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    x = np.arange(len(subjects))
    
    # Mean z-scores
    axes[0].bar(x, means, alpha=0.7, color='blue')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(subjects, rotation=45)
    axes[0].set_ylabel('Mean Z-score (positive values)')
    axes[0].set_title('Mean MD Activation')
    axes[0].grid(alpha=0.3, axis='y')
    
    # Max z-scores
    axes[1].bar(x, maxs, alpha=0.7, color='green')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(subjects, rotation=45)
    axes[1].set_ylabel('Max Z-score')
    axes[1].set_title('Peak MD Activation')
    axes[1].grid(alpha=0.3, axis='y')
    
    # Number of contrasts
    axes[2].bar(x, n_contrasts, alpha=0.7, color='orange')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(subjects, rotation=45)
    axes[2].set_ylabel('Number of Contrasts')
    axes[2].set_title('Data Availability')
    axes[2].grid(alpha=0.3, axis='y')
    
    plt.suptitle('MD System Comparison Across Subjects')
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'subject_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to: {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Print summary table
    print("\nSubject Comparison Summary:")
    print("=" * 80)
    print(f"{'Subject':<10} {'N Contrasts':<15} {'Mean Z':<15} {'Max Z':<15}")
    print("-" * 80)
    for i, subj in enumerate(subjects):
        print(f"sub-{subj:<7} {n_contrasts[i]:<15} {means[i]:<15.4f} {maxs[i]:<15.4f}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Visualize MD system maps')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Summarize command
    summarize_parser = subparsers.add_parser('summarize', help='Print summary statistics')
    summarize_parser.add_argument('map_path', help='Path to MD map')
    
    # Histogram command
    hist_parser = subparsers.add_parser('histogram', help='Plot histogram')
    hist_parser.add_argument('map_path', help='Path to MD map')
    hist_parser.add_argument('--output', help='Output path for figure')
    hist_parser.add_argument('--threshold', type=float, help='Threshold to mark')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare subjects')
    compare_parser.add_argument('md_maps_dir', help='Directory containing subject MD maps')
    compare_parser.add_argument('--output', help='Output directory for figures')
    
    args = parser.parse_args()
    
    if args.command == 'summarize':
        summarize_md_map(args.map_path)
    elif args.command == 'histogram':
        plot_md_histogram(args.map_path, args.output, args.threshold)
    elif args.command == 'compare':
        compare_subjects(args.md_maps_dir, args.output)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
