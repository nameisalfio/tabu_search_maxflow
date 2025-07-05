import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict

def _setup_style(palette: str = "viridis", context: str = "talk"):
    """
    Imposta lo stile per i grafici Seaborn.
    Il parametro font_scale è stato ridotto da 1.1 a 0.9 per rimpicciolire i testi.
    """
    sns.set_theme(style="whitegrid", palette=palette, font_scale=0.9) 
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
    
    ax.plot(iterations, values, color=line_color, linewidth=3, markersize=10, markerfacecolor='white', markeredgewidth=2.5, label='Best Solution')
    ax.fill_between(iterations, values, alpha=0.15, color=line_color)
    
    theoretical_max = results.get('theoretical_max')
    if theoretical_max:
        ax.axhline(y=theoretical_max, color='gray', linestyle='--', linewidth=2.5,
                   label=f'Theoretical Max: {theoretical_max:.2f}')

    # Le dimensioni dei font ora verranno scalate automaticamente da font_scale
    ax.set_title(f'Best Run Convergence - Network {network_number}', fontsize=22, pad=20, weight='bold')
    ax.set_xlabel('Iteration', fontsize=16, labelpad=15)
    ax.set_ylabel('Flow Value', fontsize=16, labelpad=15)
    ax.legend(fontsize=12, frameon=True, facecolor='white', framealpha=0.8)
    
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, which='both', linestyle=':', linewidth=0.7)
    sns.despine(ax=ax, offset=10)

    filename = f"convergence_best_run_network_{network_number}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Best run plot saved: {filepath}")
