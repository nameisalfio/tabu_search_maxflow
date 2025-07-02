#!/usr/bin/env python3
"""
Maximum Flow Problem - Tabu Search Implementation
Main entry point
"""

import sys
import json
import time
from pathlib import Path
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.network_reader import NetworkReader
from src.algorithms.tabu_search import TabuSearch
from src.visualization.plotter import FlowPlotter
from src.utils.config import Config

def run_single_experiment(network_file: str, config: Config, seed: int = 42):
    """Esegue un singolo esperimento"""
    # Setup
    import numpy as np
    np.random.seed(seed)
    
    # Carica network
    reader = NetworkReader()
    graph = reader.read_network_file(network_file)
    
    if graph is None:
        print(f"❌ Failed to load network: {network_file}")
        return None
    
    # Configura algoritmo
    ts_params = config.get('tabu_search', {})
    algorithm = TabuSearch(**ts_params)
    
    # Risolvi
    print(f"\n🎯 Running Tabu Search on {Path(network_file).name}...")
    solution = algorithm.solve(graph, verbose=config.get('output.verbose', True))
    
    # Risultati
    result = {
        'network_file': network_file,
        'seed': seed,
        'flow_value': solution.flow_value,
        'execution_time': algorithm.execution_time,
        'iterations': algorithm.iterations_executed,
        'convergence_history': algorithm.convergence_history,
        'parameters': ts_params
    }
    
    return result, algorithm

def run_multiple_experiments(network_file: str, config: Config):
    """Esegue esperimenti multipli"""
    seeds = config.get('experiments.random_seeds', [42])
    num_runs = len(seeds)
    
    print(f"\n🔬 Running {num_runs} experiments on {Path(network_file).name}")
    
    results = []
    convergence_data = {}
    
    for i, seed in enumerate(seeds):
        print(f"\n--- Run {i+1}/{num_runs} (seed={seed}) ---")
        
        result, algorithm = run_single_experiment(network_file, config, seed)
        if result:
            results.append(result)
            convergence_data[f"Run {i+1}"] = algorithm.convergence_history
    
    if not results:
        print("❌ No successful runs!")
        return None
    
    # Calcola statistiche
    flow_values = [r['flow_value'] for r in results]
    exec_times = [r['execution_time'] for r in results]
    iterations = [r['iterations'] for r in results]
    
    statistics = {
        'best': max(flow_values),
        'mean': sum(flow_values) / len(flow_values),
        'std': (sum((x - sum(flow_values)/len(flow_values))**2 for x in flow_values) / len(flow_values))**0.5,
        'mean_execution_time': sum(exec_times) / len(exec_times),
        'mean_iterations': sum(iterations) / len(iterations),
        'total_runs': len(results)
    }
    
    # Visualizzazione
    if config.get('visualization.save_plots', True) or config.get('visualization.show_plots', False):
        plotter = FlowPlotter(
            theme=config.get('visualization.theme', 'dark'),
            figsize=tuple(config.get('visualization.figsize', [12, 8]))
        )
        
        # Plot convergenza prima run
        output_dir = Path("data/results")
        output_dir.mkdir(exist_ok=True)
        
        plotter.plot_convergence(
            results[0]['convergence_history'],
            title=f"Convergence - {Path(network_file).name}",
            save_path=str(output_dir / f"convergence_{Path(network_file).stem}.png") if config.get('visualization.save_plots') else None,
            show=config.get('visualization.show_plots', False)
        )
        
        # Plot multiple runs se più di una
        if len(results) > 1:
            plotter.plot_multiple_runs(
                convergence_data,
                title=f"Multiple Runs - {Path(network_file).name}",
                save_path=str(output_dir / f"multiple_runs_{Path(network_file).stem}.png") if config.get('visualization.save_plots') else None,
                show=config.get('visualization.show_plots', False)
            )
    
    return {
        'network': Path(network_file).name,
        'statistics': statistics,
        'individual_results': results
    }

def main():
    """Funzione principale"""
    parser = argparse.ArgumentParser(description="Maximum Flow Tabu Search")
    parser.add_argument("--network", type=str, required=True,
                       help="Path to network file")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Config file path")
    parser.add_argument("--multiple", action="store_true",
                       help="Run multiple experiments")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for single run")
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    print("🚀 Maximum Flow Problem - Tabu Search")
    print("=" * 50)
    
    if args.multiple:
        # Multiple experiments
        final_results = run_multiple_experiments(args.network, config)
        
        if final_results and config.get('output.save_results', True):
            # Save results
            output_dir = Path("data/results")
            output_dir.mkdir(exist_ok=True)
            
            results_file = output_dir / f"results_{Path(args.network).stem}.json"
            with open(results_file, 'w') as f:
                json.dump(final_results, f, indent=2)
            
            print(f"\n💾 Results saved to: {results_file}")
            
            # Print summary
            stats = final_results['statistics']
            print(f"\n📊 SUMMARY for {final_results['network']}:")
            print(f"   Best:     {stats['best']}")
            print(f"   Mean:     {stats['mean']:.2f}")
            print(f"   Std:      {stats['std']:.2f}")
            print(f"   Avg Time: {stats['mean_execution_time']:.2f}s")
            print(f"   Avg Iter: {stats['mean_iterations']:.0f}")
    
    else:
        # Single experiment
        result, algorithm = run_single_experiment(args.network, config, args.seed)
        
        if result and config.get('output.save_results', True):
            # Save single result
            output_dir = Path("data/results")
            output_dir.mkdir(exist_ok=True)
            
            results_file = output_dir / f"single_result_{Path(args.network).stem}.json"
            with open(results_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"\n💾 Result saved to: {results_file}")
        
        # Plot convergence
        if result and (config.get('visualization.save_plots', True) or config.get('visualization.show_plots', False)):
            plotter = FlowPlotter(
                theme=config.get('visualization.theme', 'dark'),
                figsize=tuple(config.get('visualization.figsize', [12, 8]))
            )
            
            output_dir = Path("data/results")
            output_dir.mkdir(exist_ok=True)
            
            plotter.plot_convergence(
                algorithm.convergence_history,
                title=f"Tabu Search Convergence - {Path(args.network).name}",
                save_path=str(output_dir / f"convergence_{Path(args.network).stem}.png") if config.get('visualization.save_plots') else None,
                show=config.get('visualization.show_plots', False)
            )
    
    print("\n✅ Execution completed!")

if __name__ == "__main__":
    main()
