import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict

def _setup_style(palette: str = "viridis", context: str = "talk"):
    sns.set_theme(style="whitegrid", palette=palette, font_scale=1.1)
    sns.set_context(context)

def plot_best_run_convergence(results: Dict, network_number: str, output_dir: str):
    _setup_style(palette="rocket")

    valid_results = [r for r in results['run_results'] if r.get('is_valid', False)]
    if not valid_results:
        return

    best_run = max(valid_results, key=lambda x: x['best_value'])
    history = best_run.get('convergence_history')
    
    if not history:
        print("Warning: Convergence history not found for the best run.")
        return

    fig, ax = plt.subplots(figsize=(14, 8))
    
    iterations = np.array(range(1, len(history) + 1))
    values = np.array(history)
    
    line_color = sns.color_palette("rocket", 1)[0]
    
    ax.plot(iterations, values, color=line_color, linewidth=3, marker='o', markersize=10, markerfacecolor='white', markeredgewidth=2.5, label='Best Solution')
    ax.fill_between(iterations, values, alpha=0.15, color=line_color)
    
    theoretical_max = results.get('theoretical_max')
    if theoretical_max:
        ax.axhline(y=theoretical_max, color='gray', linestyle='--', linewidth=2.5,
                   label=f'Theoretical Max: {theoretical_max:.2f}')

    ax.set_title(f'Best Run Convergence - Network {network_number}', fontsize=24, pad=20, weight='bold')
    ax.set_xlabel('Iteration', fontsize=18, labelpad=15)
    ax.set_ylabel('Flow Value', fontsize=18, labelpad=15)
    ax.legend(fontsize=14, frameon=True, facecolor='white', framealpha=0.8)
    
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, which='both', linestyle=':', linewidth=0.7)
    sns.despine(ax=ax, offset=10)

    filename = f"convergence_best_run_network_{network_number}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Best run plot saved: {filepath}")


def plot_multiple_runs_convergence(results: Dict, network_number: str, output_dir: str):
    _setup_style(palette="mako")

    valid_results = [r for r in results['run_results'] if r.get('is_valid', False) and r.get('convergence_history')]
    if len(valid_results) < 2:
        return

    plot_data = []
    for run in valid_results:
        for i, value in enumerate(run['convergence_history']):
            plot_data.append({'Iteration': i + 1, 'Flow Value': value})
    
    df = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=(14, 8))
    
    line_color = sns.color_palette("mako", 1)[0]
    
    # Since std dev is 0, the confidence interval is a line. We plot it manually for better styling.
    mean_values = df.groupby('Iteration')['Flow Value'].mean()
    iterations = mean_values.index.to_numpy()
    
    ax.plot(iterations, mean_values, color=line_color, linewidth=3, marker='o', markersize=10, markerfacecolor='white', markeredgewidth=2.5, label='Mean Performance')
    
    # If there is variance, the original seaborn plot is better
    std_dev = df.groupby('Iteration')['Flow Value'].std().sum()
    if std_dev > 1e-6:
        ci_area = df.groupby('Iteration')['Flow Value'].agg(lambda x: np.percentile(x, [2.5, 97.5])).apply(pd.Series)
        ci_area = ci_area.rename(columns={0: 'low', 1: 'high'})
        ax.fill_between(ci_area.index, ci_area['low'], ci_area['high'], alpha=0.2, color=line_color, label='95% Confidence Interval')

    theoretical_max = results.get('theoretical_max')
    if theoretical_max:
        ax.axhline(y=theoretical_max, color='gray', linestyle='--', linewidth=2.5,
                   label=f'Theoretical Max: {theoretical_max:.2f}')

    ax.set_title(f'Multiple Runs Convergence - Network {network_number}', fontsize=24, pad=20, weight='bold')
    ax.set_xlabel('Iteration', fontsize=18, labelpad=15)
    ax.set_ylabel('Flow Value', fontsize=18, labelpad=15)
    ax.legend(fontsize=14, frameon=True, facecolor='white', framealpha=0.8)

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, which='both', linestyle=':', linewidth=0.7)
    sns.despine(ax=ax, offset=10)

    filename = f"convergence_multiple_runs_network_{network_number}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Multiple runs plot saved: {filepath}")
