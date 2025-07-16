import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import yaml
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

def _setup_style(style: str = "dark"):
    """Imposta uno stile grafico avanzato per matplotlib."""
    if style == "dark":
        plt.style.use('dark_background')
        plt.rcParams.update({
            "axes.facecolor": "#1a1a1a", "axes.edgecolor": "#404040",
            "axes.labelcolor": "#e0e0e0", "axes.titlecolor": "#ffffff",
            "figure.facecolor": "#0f0f0f", "grid.color": "#333333",
            "grid.alpha": 0.3, "xtick.color": "#e0e0e0",
            "ytick.color": "#e0e0e0", "text.color": "#e0e0e0",
            "legend.facecolor": "#1a1a1a", "legend.edgecolor": "#404040",
            "axes.spines.left": True, "axes.spines.bottom": True,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.linewidth": 1.2, "font.size": 11,
            "axes.labelsize": 12, "axes.titlesize": 16,
            "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10
        })
    else:
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            "axes.facecolor": "#fafafa", "axes.edgecolor": "#333333",
            "axes.labelcolor": "#2c2c2c", "axes.titlecolor": "#1a1a1a",
            "figure.facecolor": "#ffffff", "grid.color": "#cccccc",
            "grid.alpha": 0.6, "xtick.color": "#2c2c2c",
            "ytick.color": "#2c2c2c", "text.color": "#2c2c2c",
            "legend.facecolor": "#fafafa", "legend.edgecolor": "#cccccc",
            "axes.spines.left": True, "axes.spines.bottom": True,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.linewidth": 1.2, "font.size": 11,
            "axes.labelsize": 12, "axes.titlesize": 16,
            "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10
        })

def load_config(config_path='config.yaml'):
    """Carica il file di configurazione."""
    abs_path = os.path.abspath(config_path)
    print(f"Ricerca del file di configurazione in: {abs_path}")
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Attenzione: config.yaml non trovato o non valido ({e}). Uso tema light di default.")
        return {}

def parse_log_data(filepath):
    """Estrae i dati dal file di log, incluso l'ID della rete dal nome del file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()

        match_id = re.search(r'network_(\d+)', os.path.basename(filepath))
        match_nodes = re.search(r"Number of nodes:\s*(\d+)", content)
        match_edges = re.search(r"Number of edges:\s*(\d+)", content)
        match_iter = re.search(r"Mean iterations to find best solution:\s*([\d.]+)", content)
        
        if not all([match_id, match_nodes, match_edges, match_iter]):
            print(f"Attenzione: Dati mancanti nel file {filepath}")
            return None

        return {
            "network_id": int(match_id.group(1)),
            "problem_size": int(match_nodes.group(1)),
            "num_edges": int(match_edges.group(1)),
            "mean_iterations": float(match_iter.group(1)),
        }
    except (IOError, ValueError) as e:
        print(f"Errore durante il parsing del file {filepath}: {e}")
        return None

def ensure_dir_exists(path):
    """Assicura che la directory per un dato percorso esista."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

def create_categorical_scalability_plot(df, style="dark"):
    """Crea un grafico a barre categorico che confronta iterazioni e archi per istanza."""
    fig, ax1 = plt.subplots(figsize=(20, 12))

    if style == "dark":
        colors = {'iterations': '#4cc9f0', 'edges': '#f72585'}
    else:
        colors = {'iterations': '#2a9d8f', 'edges': '#e76f51'}

    x_pos = np.arange(len(df))
    bar_width = 0.35
    
    # Barre e linea per le iterazioni
    ax1.bar(x_pos - bar_width/2, df['mean_iterations'], bar_width,
            label='Mean Iterations', color=colors['iterations'], alpha=0.8)
    ax1.plot(x_pos - bar_width/2, df['mean_iterations'], color=colors['iterations'],
             marker='o', linestyle='-', linewidth=2.5, markersize=8)
             
    ax1.set_ylabel('Mean Iterations', color=colors['iterations'], fontsize=14, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=colors['iterations'], labelsize=11)
    ax1.set_xlabel('Network Instance', fontsize=14, fontweight='bold')

    # Asse secondario per gli archi
    ax2 = ax1.twinx()
    ax2.bar(x_pos + bar_width/2, df['num_edges'], bar_width,
            label='Number of Edges', color=colors['edges'], alpha=0.8)
    ax2.plot(x_pos + bar_width/2, df['num_edges'], color=colors['edges'],
             marker='o', linestyle='-', linewidth=2.5, markersize=8)

    ax2.set_ylabel('Number of Edges', color=colors['edges'], fontsize=14, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=colors['edges'], labelsize=11)

    # Configurazione asse X
    ax1.set_xticks(x_pos)
    # Usa l'ID della rete per le etichette, come richiesto
    xticklabels = [f"network_{nid}" for nid in df['network_id']]
    ax1.set_xticklabels(xticklabels, rotation=45, ha='right', fontsize=11)

    # Titolo e legenda
    plt.suptitle('Scalability Analysis per Network Instance', fontsize=22, fontweight='bold', y=0.98)
    
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=12)

    ax1.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

def create_single_correlation_plot(df, style="dark"):
    """Crea un singolo grafico di correlazione tra archi e iterazioni."""
    fig, ax = plt.subplots(figsize=(12, 8))

    if style == "dark":
        colors = {'scatter': '#4cc9f0', 'line': '#f72585'}
    else:
        colors = {'scatter': '#2a9d8f', 'line': '#e76f51'}

    # Scatter plot
    ax.scatter(df['num_edges'], df['mean_iterations'],
               s=120, c=colors['scatter'], alpha=0.7, edgecolors='white', linewidth=1.5)

    # Linea di regressione lineare
    z = np.polyfit(df['num_edges'], df['mean_iterations'], 1)
    p = np.poly1d(z)
    ax.plot(df['num_edges'], p(df['num_edges']),
            color=colors['line'], linewidth=3,
            label=f'Trend Line (y={z[0]:.4f}x + {z[1]:.2f})')

    ax.set_xlabel('Number of Edges', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Iterations', fontsize=12, fontweight='bold')
    ax.set_title('Linear Correlation: Edge Count vs. Mean Iterations', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    return fig

def main():
    """Funzione principale per eseguire l'analisi e generare i grafici."""
    config = load_config('config.yaml')
    vis_config = config.get('visualization', {})
    style = vis_config.get('style', 'dark')

    _setup_style(style)
    print(f"--- Utilizzo del tema '{style}' per i grafici ---")

    results_dir = 'data/results'
    log_files = glob.glob(os.path.join(results_dir, 'network_*', '*.log'))

    if not log_files:
        print("Nessun file di log trovato.")
        return

    performance_data = [d for d in [parse_log_data(f) for f in log_files] if d is not None]
    if not performance_data:
        print("Nessun dato valido estratto.")
        return

    # Ordina il DataFrame in base a 'network_id' per rispettare l'ordine dei file
    df = pd.DataFrame(performance_data).sort_values(by="network_id").reset_index(drop=True)

    print("\n--- Dati di Performance Estratti (ordinati per istanza) ---")
    print(df.to_string(index=False))

    # Grafico 1: Scalabilità categorica
    print("\n--- Creazione Grafico di Scalabilità per Istanza ---")
    fig1 = create_categorical_scalability_plot(df, style)
    output_path1 = os.path.join(results_dir, 'stats', 'scalability_by_instance.png')
    ensure_dir_exists(output_path1)
    fig1.savefig(output_path1, dpi=300, bbox_inches='tight', facecolor=fig1.get_facecolor())
    print(f"Grafico salvato in: {output_path1}")

    # Grafico 2: Correlazione singola
    print("\n--- Creazione Grafico di Correlazione (Archi vs Iterazioni) ---")
    fig2 = create_single_correlation_plot(df, style)
    output_path2 = os.path.join(results_dir, 'stats', 'correlation_edges_vs_iterations.png')
    ensure_dir_exists(output_path2)
    fig2.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor=fig2.get_facecolor())
    print(f"Grafico di correlazione salvato in: {output_path2}")

    plt.show()

if __name__ == "__main__":
    main()