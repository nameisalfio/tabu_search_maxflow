import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import yaml 

def _setup_style(style: str = "dark"):
    if style == "dark":
        plt.style.use('dark_background')
        plt.rcParams.update({
            "axes.facecolor": "#2b2b2b", "axes.edgecolor": "#cccccc",
            "axes.labelcolor": "white", "axes.titlecolor": "white",
            "figure.facecolor": "#1e1e1e", "grid.color": "#555555",
            "xtick.color": "white", "ytick.color": "white",
            "text.color": "white", "legend.facecolor": "#333333",
        })
    else:
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            "axes.facecolor": "white", "axes.edgecolor": "black",
            "axes.labelcolor": "black", "axes.titlecolor": "black",
            "figure.facecolor": "white", "grid.color": "#dddddd",
            "xtick.color": "black", "ytick.color": "black",
            "text.color": "black", "legend.facecolor": "white",
        })

def load_config(config_path='config.yaml'):
    abs_path = os.path.abspath(config_path)
    print(f"Searching for config file at: {abs_path}")
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError):
        print("Warning: config.yaml not found or invalid. Defaulting to light theme.")
        return {}

def parse_log_data(filepath):
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        match_size = re.search(r'network_(\d+)', filepath)
        match_iter = re.search(r"Mean iterations to find best solution:\s*([\d.]+)", content)
        match_time = re.search(r"Mean execution time per run:\s*([\d.]+) seconds", content)
        match_edges = re.search(r"Number of edges:\s*(\d+)", content)
        if not all([match_size, match_iter, match_time, match_edges]): return None
        return {
            "problem_size": int(match_size.group(1)),
            "mean_iterations": float(match_iter.group(1)),
            "mean_time": float(match_time.group(1)),
            "num_edges": int(match_edges.group(1))
        }
    except (IOError, ValueError):
        return None

def main():
    config = load_config('config.yaml')
    vis_config = config.get('visualization', {})
    style = vis_config.get('style', 'light')
    
    _setup_style(style)
    sns.set_theme(style='darkgrid' if style == 'dark' else 'whitegrid')
    palette = sns.color_palette("viridis", 2)
    color_iter, color_time = palette[0], palette[1]
    
    print(f"--- Using '{style}' theme for plots ---")

    results_dir = 'data/results'
    log_files = glob.glob(os.path.join(results_dir, 'network_*', '*.log'))

    if not log_files:
        print("No log files found.")
        return

    performance_data = [d for d in [parse_log_data(f) for f in log_files] if d is not None]
    if not performance_data:
        print("No valid data extracted.")
        return

    df = pd.DataFrame(performance_data).sort_values(by="problem_size").reset_index(drop=True)
    df['problem_size_str'] = df['problem_size'].astype(str)
    print("\n--- Extracted Performance Data ---")
    print(df)

    fig, ax1 = plt.subplots(figsize=(18, 10))
    bar_width = 0.35
    x_pos = np.arange(len(df['problem_size_str']))

    bars1 = ax1.bar(x_pos - bar_width/2, df['mean_iterations'], bar_width, label='Mean Iterations', color=color_iter, alpha=0.7)
    ax1.plot(x_pos - bar_width/2, df['mean_iterations'], color=color_iter, marker='o', linestyle='-')
    ax1.set_ylabel('Mean Iterations', color=color_iter, fontsize=14)
    ax1.tick_params(axis='y', labelcolor=color_iter)
    ax1.set_xlabel('Network Size', fontsize=14)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(df['problem_size_str'])
    ax1.tick_params(axis='x', rotation=45)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x_pos + bar_width/2, df['mean_time'], bar_width, label='Mean Execution Time (s)', color=color_time, alpha=0.7)
    ax2.plot(x_pos + bar_width/2, df['mean_time'], color=color_time, marker='s', linestyle='-')
    ax2.set_ylabel('Mean Execution Time (s)', color=color_time, fontsize=14)
    ax2.tick_params(axis='y', labelcolor=color_time)

    plt.title('Algorithm Scalability: Iterations and Execution Time', fontsize=18, fontweight='bold')
    fig.legend(handles=[bars1, bars2], loc='upper left', bbox_to_anchor=(0.1, 0.92))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = os.path.join(results_dir, 'stats', f'time_vs_iter.png')
    plt.savefig(output_path, dpi=300)
    print(f"\nSaved grouped plot to: {output_path}")
    plt.show()

    fig2, ax3 = plt.subplots(figsize=(18, 10))
    x_pos_edges = np.arange(len(df))
    df_sorted_edges = df.sort_values(by='num_edges').reset_index(drop=True)
    bar_width = 0.35

    bars3 = ax3.bar(x_pos_edges - bar_width/2, df_sorted_edges['mean_iterations'], bar_width, label='Mean Iterations', color=color_iter, alpha=0.7)
    ax3.plot(x_pos_edges - bar_width/2, df_sorted_edges['mean_iterations'], color=color_iter, marker='o', linestyle='-')
    ax3.set_ylabel('Mean Iterations', color=color_iter, fontsize=14)
    ax3.tick_params(axis='y', labelcolor=color_iter)
    ax3.set_xlabel('Number of Edges', fontsize=14)
    ax3.set_xticks(x_pos_edges)
    ax3.set_xticklabels(df_sorted_edges['num_edges'].astype(str), rotation=45)

    ax4 = ax3.twinx()
    bars4 = ax4.bar(x_pos_edges + bar_width/2, df_sorted_edges['mean_time'], bar_width, label='Mean Execution Time (s)', color=color_time, alpha=0.7)
    ax4.plot(x_pos_edges + bar_width/2, df_sorted_edges['mean_time'], color=color_time, marker='s', linestyle='-')
    ax4.set_ylabel('Mean Execution Time (s)', color=color_time, fontsize=14)
    ax4.tick_params(axis='y', labelcolor=color_time)

    plt.title('Scalability vs. Number of Edges', fontsize=18, fontweight='bold')
    fig2.legend(handles=[bars3, bars4], loc='upper left', bbox_to_anchor=(0.1, 0.92))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path_edges = os.path.join(results_dir, 'stats', f'edges_vs_iter_time.png')
    plt.savefig(output_path_edges, dpi=300)
    print(f"Saved edges-based plot to: {output_path_edges}")
    plt.show()

if __name__ == "__main__":
    main()
