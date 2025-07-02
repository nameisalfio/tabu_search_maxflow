"""
Visualization Module for Maximum Flow Results
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path

class FlowPlotter:
    """Plotter per risultati Maximum Flow con tema scuro"""
    
    def __init__(self, theme="dark", figsize=(12, 8)):
        self.theme = theme
        self.figsize = figsize
        self._setup_style()
    
    def _setup_style(self):
        """Configura stile matplotlib"""
        if self.theme == "dark":
            plt.style.use('dark_background')
            self.colors = {
                'primary': '#00d4ff',
                'secondary': '#ff6b35', 
                'accent': '#7fff00',
                'text': '#ffffff',
                'grid': '#404040'
            }
        else:
            plt.style.use('default')
            self.colors = {
                'primary': '#1f77b4',
                'secondary': '#ff7f0e',
                'accent': '#2ca02c', 
                'text': '#000000',
                'grid': '#cccccc'
            }
    
    def plot_convergence(self, convergence_history: List[float], 
                        title: str = "Tabu Search Convergence",
                        save_path: Optional[str] = None,
                        show: bool = False) -> plt.Figure:
        """Plotta convergenza algoritmo"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        iterations = range(len(convergence_history))
        
        # Plot principale
        ax.plot(iterations, convergence_history, 
               color=self.colors['primary'], linewidth=2.5, alpha=0.9)
        
        # Evidenzia valore finale
        final_value = convergence_history[-1]
        ax.axhline(y=final_value, color=self.colors['secondary'], 
                  linestyle='--', alpha=0.7, linewidth=2)
        
        # Annotazione valore finale
        ax.annotate(f'Final: {final_value}', 
                   xy=(len(iterations) * 0.7, final_value),
                   xytext=(len(iterations) * 0.7, final_value * 1.1),
                   color=self.colors['secondary'],
                   fontsize=12, fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color=self.colors['secondary']))
        
        # Styling
        ax.set_xlabel('Iterations', fontsize=14, color=self.colors['text'])
        ax.set_ylabel('Flow Value', fontsize=14, color=self.colors['text'])
        ax.set_title(title, fontsize=16, fontweight='bold', color=self.colors['text'])
        ax.grid(True, alpha=0.3, color=self.colors['grid'])
        
        # Miglioramenti estetici
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if self.theme == "dark":
            ax.spines['bottom'].set_color(self.colors['grid'])
            ax.spines['left'].set_color(self.colors['grid'])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='black' if self.theme == "dark" else 'white')
            print(f"📊 Plot saved: {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_multiple_runs(self, runs_data: Dict[str, List[float]],
                          title: str = "Multiple Runs Comparison",
                          save_path: Optional[str] = None,
                          show: bool = False) -> plt.Figure:
        """Plotta confronto multiple run"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(runs_data)))
        
        for i, (run_name, convergence) in enumerate(runs_data.items()):
            iterations = range(len(convergence))
            ax.plot(iterations, convergence, 
                   color=colors[i], linewidth=2, alpha=0.8, label=run_name)
        
        ax.set_xlabel('Iterations', fontsize=14, color=self.colors['text'])
        ax.set_ylabel('Flow Value', fontsize=14, color=self.colors['text'])
        ax.set_title(title, fontsize=16, fontweight='bold', color=self.colors['text'])
        ax.grid(True, alpha=0.3, color=self.colors['grid'])
        ax.legend(framealpha=0.9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='black' if self.theme == "dark" else 'white')
            print(f"📊 Plot saved: {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_statistics_summary(self, results: Dict[str, Dict],
                               save_path: Optional[str] = None,
                               show: bool = False) -> plt.Figure:
        """Plotta summary statistiche per network diversi"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        networks = list(results.keys())
        best_values = [results[net]['best'] for net in networks]
        mean_values = [results[net]['mean'] for net in networks]
        std_values = [results[net]['std'] for net in networks]
        exec_times = [results[net]['execution_time'] for net in networks]
        
        # Plot 1: Best vs Mean
        x = np.arange(len(networks))
        width = 0.35
        
        ax1.bar(x - width/2, best_values, width, label='Best', 
               color=self.colors['primary'], alpha=0.8)
        ax1.bar(x + width/2, mean_values, width, label='Mean', 
               color=self.colors['secondary'], alpha=0.8)
        
        ax1.set_xlabel('Network', color=self.colors['text'])
        ax1.set_ylabel('Flow Value', color=self.colors['text'])
        ax1.set_title('Best vs Mean Flow Values', fontweight='bold', color=self.colors['text'])
        ax1.set_xticks(x)
        ax1.set_xticklabels(networks, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Mean with error bars
        ax2.errorbar(x, mean_values, yerr=std_values, fmt='o-', 
                    capsize=5, color=self.colors['accent'], linewidth=2)
        ax2.set_xlabel('Network', color=self.colors['text'])
        ax2.set_ylabel('Flow Value (Mean ± Std)', color=self.colors['text'])
        ax2.set_title('Mean with Standard Deviation', fontweight='bold', color=self.colors['text'])
        ax2.set_xticks(x)
        ax2.set_xticklabels(networks, rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Execution times
        bars = ax3.bar(x, exec_times, color=self.colors['primary'], alpha=0.8)
        ax3.set_xlabel('Network', color=self.colors['text'])
        ax3.set_ylabel('Execution Time (s)', color=self.colors['text'])
        ax3.set_title('Execution Times', fontweight='bold', color=self.colors['text'])
        ax3.set_xticks(x)
        ax3.set_xticklabels(networks, rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Aggiungi valori sopra le barre
        for bar, time_val in zip(bars, exec_times):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{time_val:.1f}s', ha='center', va='bottom', 
                    color=self.colors['text'], fontsize=10)
        
        # Plot 4: Coefficiente di variazione
        cv_values = [std_values[i]/mean_values[i] if mean_values[i] > 0 else 0 
                    for i in range(len(networks))]
        
        ax4.bar(x, cv_values, color=self.colors['accent'], alpha=0.8)
        ax4.set_xlabel('Network', color=self.colors['text'])
        ax4.set_ylabel('Coefficient of Variation', color=self.colors['text'])
        ax4.set_title('Solution Stability (CV)', fontweight='bold', color=self.colors['text'])
        ax4.set_xticks(x)
        ax4.set_xticklabels(networks, rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='black' if self.theme == "dark" else 'white')
            print(f"📊 Statistics plot saved: {save_path}")
        
        if show:
            plt.show()
        
        return fig
