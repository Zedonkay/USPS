#!/usr/bin/env python3
"""
Visualization script for test results from robust testing.

This script reads JSON test result files and creates various visualizations
to analyze the performance across different parameter perturbations.
"""

import os
import json
import argparse
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def parse_filename(filename: str) -> Tuple[str, float]:
    """
    Parse a test result filename to extract parameter name and value.
    
    Format: {param}-constant-{value}.json
    Example: actuator_kp-constant-1.011e+00.json
    
    Returns:
        Tuple of (parameter_name, parameter_value)
    """
    match = re.match(r'(.+)-constant-([\d.]+e[+-]\d+)\.json', filename)
    if match:
        param_name = match.group(1)
        param_value = float(match.group(2))
        return param_name, param_value
    return None, None


def load_test_results(test_dir: str) -> Dict[str, List[Dict]]:
    """
    Load all test result JSON files from the test directory.
    
    Returns:
        Dictionary mapping parameter names to lists of test results.
        Each result dict contains: value, mean_reward, std_reward, min_reward,
        episode_rewards, episode_lengths, perturb_spec
    """
    results = defaultdict(list)
    test_path = Path(test_dir)
    
    if not test_path.exists():
        raise ValueError(f"Test directory does not exist: {test_dir}")
    
    for json_file in test_path.glob("*.json"):
        param_name, param_value = parse_filename(json_file.name)
        if param_name is None:
            print(f"Warning: Could not parse filename {json_file.name}, skipping...")
            continue
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            result = {
                'value': param_value,
                'mean_reward': data['mean_reward'],
                'std_reward': data['std_reward'],
                'min_reward': data['min_reward'],
                'episode_rewards': data['episode_rewards'],
                'episode_lengths': data['episode_lengths'],
                'perturb_spec': data['perturb_spec']
            }
            results[param_name].append(result)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue
    
    # Sort each parameter's results by value
    for param_name in results:
        results[param_name].sort(key=lambda x: x['value'])
    
    return results


def plot_mean_reward_vs_param(results: Dict[str, List[Dict]], output_dir: str):
    """Plot mean reward vs parameter value for each parameter type.
    Creates separate plots for each parameter with shaded regions."""
    n_params = len(results)
    if n_params == 0:
        print("No results to plot")
        return
    
    # Create images directory
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    # Create a separate plot for each parameter
    for param_name, param_results in sorted(results.items()):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        values = [r['value'] for r in param_results]
        mean_rewards = [r['mean_reward'] for r in param_results]
        std_rewards = [r['std_reward'] for r in param_results]
        
        # Plot mean line
        ax.plot(values, mean_rewards, marker='o', linewidth=2.5, 
               markersize=8, label='Mean Reward', color='#2E86AB')
        
        # Shade the region between mean ± std
        upper_bound = [m + s for m, s in zip(mean_rewards, std_rewards)]
        lower_bound = [m - s for m, s in zip(mean_rewards, std_rewards)]
        ax.fill_between(values, lower_bound, upper_bound, 
                       alpha=0.3, color='#2E86AB', 
                       label='Mean ± Std')
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_xlabel('Parameter Value', fontsize=12)
        ax.set_ylabel('Mean Reward', fontsize=12)
        ax.set_title(f'{param_name.replace("_", " ").title()}', 
                     fontsize=14, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=11)
        
        plt.tight_layout()
        # Save with sanitized filename
        safe_param_name = param_name.replace('_', '_')
        output_path = os.path.join(images_dir, f'{safe_param_name}_mean_reward.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()


def plot_combined_all_params(results: Dict[str, List[Dict]], output_dir: str):
    """Plot all parameters combined in a single plot with normalized x-axis.
    Each parameter is normalized to 0-1 range for comparison."""
    n_params = len(results)
    if n_params == 0:
        print("No results to plot")
        return
    
    # Create images directory
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    # Define a color palette
    colors = plt.cm.tab10(np.linspace(0, 1, n_params))
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for idx, (param_name, param_results) in enumerate(sorted(results.items())):
        values = [r['value'] for r in param_results]
        mean_rewards = [r['mean_reward'] for r in param_results]
        std_rewards = [r['std_reward'] for r in param_results]
        
        # Normalize values to 0-1 range for comparison
        if len(values) > 1:
            min_val, max_val = min(values), max(values)
            normalized_values = [(v - min_val) / (max_val - min_val) if max_val != min_val else 0.5 
                               for v in values]
        else:
            normalized_values = [0.5]
        
        # Get color for this parameter
        color = colors[idx]
        
        # Plot mean line
        ax.plot(normalized_values, mean_rewards, marker='o', linewidth=2.5, 
               markersize=6, label=param_name.replace('_', ' ').title(), 
               color=color, alpha=0.8)
        
        # Shade the region between mean ± std
        upper_bound = [m + s for m, s in zip(mean_rewards, std_rewards)]
        lower_bound = [m - s for m, s in zip(mean_rewards, std_rewards)]
        ax.fill_between(normalized_values, lower_bound, upper_bound, 
                       alpha=0.2, color=color)
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Normalized Parameter Value (0 = min, 1 = max)', fontsize=12)
    ax.set_ylabel('Mean Reward', fontsize=12)
    ax.set_title('All Parameters - Mean Reward Comparison', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10, ncol=2, framealpha=0.9)
    
    plt.tight_layout()
    output_path = os.path.join(images_dir, 'all_params_combined.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_reward_distributions(results: Dict[str, List[Dict]], output_dir: str):
    """Plot box plots showing reward distributions for each parameter."""
    n_params = len(results)
    if n_params == 0:
        return
    
    cols = 3
    rows = (n_params + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if n_params == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (param_name, param_results) in enumerate(sorted(results.items())):
        ax = axes[idx]
        
        # Prepare data for box plot
        all_rewards = []
        labels = []
        positions = []
        
        for i, result in enumerate(param_results):
            all_rewards.append(result['episode_rewards'])
            # Use abbreviated value labels
            value_str = f"{result['value']:.3f}"
            labels.append(value_str)
            positions.append(i)
        
        bp = ax.boxplot(all_rewards, labels=labels, patch_artist=True)
        
        # Color the boxes
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Parameter Value', fontsize=11)
        ax.set_ylabel('Episode Reward', fontsize=11)
        ax.set_title(f'{param_name.replace("_", " ").title()} - Reward Distribution', 
                     fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
    
    # Hide unused subplots
    for idx in range(n_params, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'reward_distributions.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_min_reward_vs_param(results: Dict[str, List[Dict]], output_dir: str):
    """Plot minimum reward vs parameter value."""
    n_params = len(results)
    if n_params == 0:
        return
    
    cols = 3
    rows = (n_params + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if n_params == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (param_name, param_results) in enumerate(sorted(results.items())):
        ax = axes[idx]
        
        values = [r['value'] for r in param_results]
        min_rewards = [r['min_reward'] for r in param_results]
        mean_rewards = [r['mean_reward'] for r in param_results]
        
        ax.plot(values, min_rewards, marker='s', linewidth=2, 
               label='Min Reward', color='red', markersize=8)
        ax.plot(values, mean_rewards, marker='o', linewidth=2, 
               label='Mean Reward', color='blue', markersize=6, alpha=0.7)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Parameter Value', fontsize=11)
        ax.set_ylabel('Reward', fontsize=11)
        ax.set_title(f'{param_name.replace("_", " ").title()} - Min vs Mean', 
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Hide unused subplots
    for idx in range(n_params, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'min_reward_vs_param.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_summary_statistics(results: Dict[str, List[Dict]], output_dir: str):
    """Create a summary statistics table visualization."""
    fig, ax = plt.subplots(figsize=(14, max(6, len(results) * 0.5)))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    headers = ['Parameter', 'Value Range', 'Best Mean', 'Worst Mean', 
               'Best Min', 'Worst Min', 'Mean Std']
    
    for param_name in sorted(results.keys()):
        param_results = results[param_name]
        values = [r['value'] for r in param_results]
        mean_rewards = [r['mean_reward'] for r in param_results]
        min_rewards = [r['min_reward'] for r in param_results]
        std_rewards = [r['std_reward'] for r in param_results]
        
        best_mean_idx = np.argmax(mean_rewards)
        worst_mean_idx = np.argmin(mean_rewards)
        best_min_idx = np.argmax(min_rewards)
        worst_min_idx = np.argmin(min_rewards)
        
        row = [
            param_name.replace('_', ' ').title(),
            f"{min(values):.3f} - {max(values):.3f}",
            f"{mean_rewards[best_mean_idx]:.2f} (val={values[best_mean_idx]:.3f})",
            f"{mean_rewards[worst_mean_idx]:.2f} (val={values[worst_mean_idx]:.3f})",
            f"{min_rewards[best_min_idx]:.2f} (val={values[best_min_idx]:.3f})",
            f"{min_rewards[worst_min_idx]:.2f} (val={values[worst_min_idx]:.3f})",
            f"{np.mean(std_rewards):.2f}"
        ]
        table_data.append(row)
    
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='left', loc='center',
                    colWidths=[0.15, 0.12, 0.18, 0.18, 0.18, 0.18, 0.10])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style the header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Summary Statistics Across All Parameters', 
              fontsize=14, fontweight='bold', pad=20)
    output_path = os.path.join(output_dir, 'summary_statistics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_heatmap(results: Dict[str, List[Dict]], output_dir: str):
    """Create a heatmap showing mean rewards across parameters."""
    # Prepare data for heatmap
    param_names = sorted(results.keys())
    max_tests = max(len(results[p]) for p in param_names)
    
    # Create a matrix where rows are parameters and columns are test indices
    heatmap_data = []
    param_labels = []
    
    for param_name in param_names:
        param_results = results[param_name]
        param_labels.append(param_name.replace('_', ' ').title())
        
        # Normalize to same length by interpolating or padding
        mean_rewards = [r['mean_reward'] for r in param_results]
        values = [r['value'] for r in param_results]
        
        # Normalize values to 0-1 range for display
        if len(values) > 1:
            normalized_values = [(v - min(values)) / (max(values) - min(values)) 
                               for v in values]
        else:
            normalized_values = [0.5]
        
        row = []
        for i in range(max_tests):
            if i < len(mean_rewards):
                row.append(mean_rewards[i])
            else:
                row.append(np.nan)
        heatmap_data.append(row)
    
    heatmap_data = np.array(heatmap_data)
    
    fig, ax = plt.subplots(figsize=(max(10, max_tests * 0.5), max(6, len(param_names) * 0.6)))
    
    im = ax.imshow(heatmap_data, aspect='auto', cmap='RdYlGn', 
                   interpolation='nearest')
    
    ax.set_xticks(range(max_tests))
    ax.set_xticklabels([f'Test {i+1}' for i in range(max_tests)])
    ax.set_yticks(range(len(param_names)))
    ax.set_yticklabels(param_labels)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean Reward', rotation=270, labelpad=20)
    
    # Add text annotations
    for i in range(len(param_names)):
        for j in range(max_tests):
            if not np.isnan(heatmap_data[i, j]):
                text = ax.text(j, i, f'{heatmap_data[i, j]:.1f}',
                             ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title('Mean Reward Heatmap Across Parameters', 
                fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'reward_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize test results from robust testing experiments'
    )
    parser.add_argument(
        '--test_dir',
        type=str,
        required=True,
        help='Directory containing test result JSON files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory to save visualization plots (default: same as test_dir)'
    )
    parser.add_argument(
        '--plots',
        type=str,
        nargs='+',
        choices=['all', 'mean', 'distribution', 'min', 'summary', 'heatmap', 'combined'],
        default=['all'],
        help='Which plots to generate (default: all)'
    )
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = args.test_dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    print(f"Loading test results from: {args.test_dir}")
    results = load_test_results(args.test_dir)
    
    if len(results) == 0:
        print("No valid test results found!")
        return
    
    print(f"Found results for {len(results)} parameters:")
    for param_name in sorted(results.keys()):
        print(f"  - {param_name}: {len(results[param_name])} test cases")
    
    # Generate plots
    plots_to_generate = args.plots
    if 'all' in plots_to_generate:
        plots_to_generate = ['mean', 'distribution', 'min', 'summary', 'heatmap', 'combined']
    
    print("\nGenerating visualizations...")
    
    if 'mean' in plots_to_generate:
        plot_mean_reward_vs_param(results, args.output_dir)
    
    if 'distribution' in plots_to_generate:
        plot_reward_distributions(results, args.output_dir)
    
    if 'min' in plots_to_generate:
        plot_min_reward_vs_param(results, args.output_dir)
    
    if 'summary' in plots_to_generate:
        plot_summary_statistics(results, args.output_dir)
    
    if 'heatmap' in plots_to_generate:
        plot_heatmap(results, args.output_dir)
    
    if 'combined' in plots_to_generate:
        plot_combined_all_params(results, args.output_dir)
    
    print(f"\nAll visualizations saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
