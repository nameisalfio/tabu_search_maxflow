import os
import sys
import json
import time
import yaml
import statistics
import argparse

sys.path.append('src')

from src.data.network_reader import *
from src.algorithms.tabu_search import *
from src.visualization.plotter import *

def load_config(config_path: str = "config.yaml") -> dict:
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise Exception(f"Error loading config from {config_path}: {e}")


def run_single_experiment(network_data, config: dict, run_id: int) -> dict:
    print(f"\n\nRun {run_id + 1}: ", end="")
    
    seeds = config['experiments']['random_seeds']
    seed = seeds[run_id] if run_id < len(seeds) else (seeds[0] + run_id)
    
    import random
    random.seed(seed)
    
    tabu_search = TabuSearchMaxFlow(
        graph=network_data.graph,
        source=network_data.source,
        sink=network_data.sink,
        config_path="config.yaml"
    )
    
    start_time = time.time()
    best_flow_dict, best_value, convergence_history = tabu_search.solve()
    execution_time = time.time() - start_time
    
    is_valid, flow_value, errors = validate_flow(network_data, best_flow_dict)
    is_optimal = network_data.is_optimal_flow(flow_value)
    
    iterations_to_best = len(convergence_history)
    if convergence_history:
        for i, value in enumerate(convergence_history):
            if abs(value - best_value) < 1e-9:
                iterations_to_best = i + 1
                break
    
    result = {
        'run_id': run_id,
        'seed': seed,
        'best_value': flow_value,
        'is_valid': is_valid,
        'is_optimal': is_optimal,
        'execution_time': execution_time,
        'iterations_to_best': iterations_to_best,
        'total_iterations': tabu_search.iteration,
        'function_evaluations': tabu_search.get_function_evaluations(),
        'convergence_history': convergence_history
    }
    
    optimal_str = " (optimal)" if is_optimal else ""
    print(f"flow = {flow_value:.2f}{optimal_str}")
    
    return result


def create_plots(results: dict, network_number: str, config: dict):
    if not config['visualization']['save_plots']:
        return
    
    valid_results = [r for r in results['run_results'] if r.get('is_valid', False)]
    if not valid_results:
        print("No valid results to plot.")
        return
    
    output_dir = f"data/results/network_{network_number}"
    os.makedirs(output_dir, exist_ok=True)
    
    plot_best_run_convergence(results, network_number, output_dir)
    
    if len(valid_results) > 1:
        plot_multiple_runs_convergence(results, network_number, output_dir)


def main():
    parser = argparse.ArgumentParser(description='Tabu Search for Maximum Flow Problem')
    parser.add_argument('--network', type=str, required=True, 
                       help='Network file to process (e.g., data/networks/network_160.txt)')
    args = parser.parse_args()
    
    try:
        config = load_config("config.yaml")
        print("\nConfiguration loaded")
        print(f"Runs per experiment: {config['experiments']['num_runs']}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return
    
    if not os.path.exists(args.network):
        print(f"Error: Network file {args.network} not found")
        return
    
    print(f"Processing network: {args.network}")
    
    # Load network
    try:
        network_data = read_network(args.network, verbose=True)
    except Exception as e:
        print(f"Error loading network: {e}")
        return
    
    # Run experiments
    run_results = []
    num_runs = config['experiments']['num_runs']
    
    for run_id in range(num_runs):
        try:
            result = run_single_experiment(network_data, config, run_id)
            run_results.append(result)
        except Exception as e:
            print(f"Run {run_id + 1}: Error - {e}")
            run_results.append({
                'run_id': run_id,
                'error': str(e),
                'best_value': 0.0,
                'is_valid': False
            })
    
    # Calculate statistics
    valid_results = [r for r in run_results if r.get('is_valid', False)]
    
    if not valid_results:
        print("No valid results obtained")
        return
    
    best_values = [r['best_value'] for r in valid_results]
    iterations_to_best = [r['iterations_to_best'] for r in valid_results]
    execution_times = [r['execution_time'] for r in valid_results]
    
    aggregate_stats = {
        'best': max(best_values),
        'mean': statistics.mean(best_values),
        'std_dev': statistics.stdev(best_values) if len(best_values) > 1 else 0.0,
        'mean_iterations_to_best': statistics.mean(iterations_to_best),
        'mean_execution_time': statistics.mean(execution_times),
        'optimal_runs': sum(1 for r in valid_results if r.get('is_optimal', False))
    }
    
    theoretical_max = network_data.get_max_possible_flow()

    # Save results
    network_number = os.path.basename(args.network).replace('network_', '').replace('.txt', '')
    output_dir = f"data/results/network_{network_number}"
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'network_file': args.network,
        'network_info': network_data.info,
        'run_results': run_results,
        'aggregate_stats': aggregate_stats,
        'theoretical_max': theoretical_max
    }
    
    if config['output']['save_results']:
        # Clean results for JSON
        clean_results = results.copy()
        for run_result in clean_results.get('run_results', []):
            if 'final_flow' in run_result:
                del run_result['final_flow']  # Too large for JSON
        
        filename = f"results_network_{network_number}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(clean_results, f, indent=2)
    
    
    # Final summary
    print(f"\nEXPERIMENT SUMMARY")
    print("=" * 50)
    print(f"Best flow value found: {aggregate_stats['best']:.2f}")
    print(f"Mean of best values: {aggregate_stats['mean']:.2f}")
    print(f"Standard deviation: {aggregate_stats['std_dev']:.2f}")
    print(f"Average iterations to best: {aggregate_stats['mean_iterations_to_best']:.1f}")
    avg_function_evals = statistics.mean([r['function_evaluations'] for r in valid_results])
    print(f"Average function evaluations to best: {avg_function_evals:.1f}\n")
    print("=" * 50, end="\n\n")
    
    # Create plots
    create_plots(results, network_number, config)

if __name__ == "__main__":
    main()