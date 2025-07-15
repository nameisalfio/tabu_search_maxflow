# src/visualization/plotter.py
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Any

def _setup_style(style: str = "dark"):
    """Sets a custom plot style based on the provided parameter."""
    if style == "dark":
        plt.style.use('dark_background')
        plt.rcParams.update({
            "axes.facecolor": "#2b2b2b", "axes.edgecolor": "#cccccc",
            "axes.labelcolor": "white", "axes.titlecolor": "white",
            "figure.facecolor": "#1e1e1e", "grid.color": "#555555",
            "xtick.color": "white", "ytick.color": "white",
            "text.color": "white", "legend.facecolor": "#333333",
        })
        return {"bbox_face": "black", "bbox_edge": "white", "best_color": "#00ff7f", "marker_color": "white"}
    else:  # Default to light style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            "axes.facecolor": "white", "axes.edgecolor": "black",
            "axes.labelcolor": "black", "axes.titlecolor": "black",
            "figure.facecolor": "white", "grid.color": "#dddddd",
            "xtick.color": "black", "ytick.color": "black",
            "text.color": "black", "legend.facecolor": "white",
        })
        return {"bbox_face": "whitesmoke", "bbox_edge": "black", "best_color": "green", "marker_color": "black"}

def plot_all_runs_convergence(all_histories: List[List[float]], instance_name: str, output_dir: str, style: str, summary_stats: Dict[str, Any]):
    """
    Plots each run in a distinct color and displays a summary statistics box.
    """
    colors = _setup_style(style)
    fig, ax = plt.subplots(figsize=(16, 9))

    plot_data = []
    max_len = max(len(h) for h in all_histories) if all_histories else 0
    for i, history in enumerate(all_histories):
        padded_history = history + [history[-1]] * (max_len - len(history))
        for iteration, value in enumerate(padded_history):
            plot_data.append({"Run": f"Run {i+1}", "Iteration": iteration, "Flow Value": value})
    
    df = pd.DataFrame(plot_data)

    # Plot each run with a unique color from a palette
    sns.lineplot(data=df, x="Iteration", y="Flow Value", hue="Run",
                 palette="viridis", alpha=0.8, linewidth=1.5, ax=ax, legend=False)

    ax.set_title(f"All Runs Convergence for {instance_name}", fontsize=22, pad=20)
    ax.set_xlabel("Iteration", fontsize=16)
    ax.set_ylabel("Total Flow Value", fontsize=16)
    
    # --- Create and add the statistics text box ---
    stats_text = (
        f"PERFORMANCE SUMMARY\n"
        f"----------------------------------\n"
        f"Best Flow Found: {summary_stats['best_flow_found']:.4f}\n"
        f"Mean Flow: {summary_stats['mean_flow']:.4f}\n"
        f"Std. Deviation: {summary_stats['std_dev_flow']:.4f}\n"
        f"Success Rate: {summary_stats['success_rate_perc']:.1f}%\n"
        f"Avg. Iterations to Best: {summary_stats['mean_iterations_to_best']:.1f}\n"
        f"Avg. Evaluations to Best: {summary_stats['mean_evaluations_to_best']:.1f}"
    )
    
    props = dict(boxstyle='round,pad=0.5', facecolor=colors['bbox_face'], edgecolor=colors['bbox_edge'], alpha=0.8)
    # Position the text box in the bottom right corner
    ax.text(0.97, 0.03, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom', horizontalalignment='right', bbox=props,
            fontfamily='monospace')

    filepath = os.path.join(output_dir, f"convergence_all_runs_{instance_name}.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return filepath

def plot_best_run_convergence(best_history: List[float], instance_name: str, output_dir: str, style: str):
    """Plots the single best run with enhanced visualization."""
    colors = _setup_style(style)
    fig, ax = plt.subplots(figsize=(15, 8))

    x_data = range(len(best_history))
    y_data = best_history

    ax.plot(x_data, y_data, color=colors['best_color'], linewidth=3, label="Best Run Performance")
    ax.fill_between(x_data, y_data, color=colors['best_color'], alpha=0.2)
    
    final_iter = len(y_data) - 1
    final_flow = y_data[-1]
    ax.scatter(final_iter, final_flow, color=colors['marker_color'], s=150, zorder=5, 
               edgecolor='grey', label=f"Final Best: {final_flow:.4f}")
    
    ax.set_title(f"Best Run Convergence for {instance_name}", fontsize=20, pad=20)
    ax.set_xlabel("Iteration", fontsize=14)
    ax.set_ylabel("Best Flow Value Found", fontsize=14)
    ax.legend(fontsize=12)

    filepath = os.path.join(output_dir, f"convergence_best_run_{instance_name}.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return filepath