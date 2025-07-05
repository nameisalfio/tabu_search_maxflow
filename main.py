import os
import sys
import json
import time
import yaml
import statistics
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

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
    # Generate diverse seeds to avoid deterministic behavior
    base_seeds = config['experiments']['random_seeds']
    base_seed = base_seeds[run_id % len(base_seeds)]
    
    # Add time-based randomization to create truly different runs
    import time
    diverse_seed = base_seed + run_id * 1000 + int(time.time() * 1000) % 10000
    
    random.seed(diverse_seed)
    import numpy as np
    np.random.seed(diverse_seed)
    
    tabu_search = TabuSearch(
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
        'seed': diverse_seed,
        'best_value': flow_value,
        'is_valid': is_valid,
        'is_optimal': is_optimal,
        'execution_time': execution_time,
        'iterations_to_best': iterations_to_best,
        'total_iterations': tabu_search.iteration,
        'function_evaluations': tabu_search.get_function_evaluations(),
        'convergence_history': convergence_history
    }
    
    return result


def run_single_experiment_worker(args) -> dict:
    """Worker function for parallel execution"""
    network_data, config, run_id = args
    return run_single_experiment(network_data, config, run_id)


def run_parallel_experiments(network_data, config: dict) -> list:
    """Run experiments in parallel using multiple CPU cores"""
    num_runs = config['experiments']['num_runs']
    max_workers = min(num_runs, mp.cpu_count())
    
    print(f"Running {num_runs} experiments in parallel using {max_workers} CPU core(s)")
    
    # Prepare arguments for parallel execution
    args_list = [(network_data, config, run_id) for run_id in range(num_runs)]
    
    # Execute in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(run_single_experiment_worker, args_list))
    
    return results


def run_sequential_experiments(network_data, config: dict) -> list:
    """Run experiments sequentially"""
    num_runs = config['experiments']['num_runs']
    run_results = []
    
    print(f"Running {num_runs} experiments sequentially...")
    
    for run_id in range(num_runs):
        try:
            print(f"Run {run_id + 1:2d}/{num_runs}: ", end="", flush=True)
            
            result = run_single_experiment(network_data, config, run_id)
            run_results.append(result)
            
            flow_val = result['best_value']
            exec_time = result['execution_time']
            iterations = result['total_iterations']
            optimal = "OPT" if result['is_optimal'] else "SUB"
            
            print(f"flow={flow_val:.2f} {optimal} iter={iterations} ({exec_time:.1f}s)")
            
        except Exception as e:
            print(f"FAILED: {e}")
            run_results.append({
                'run_id': run_id,
                'error': str(e),
                'best_value': 0.0,
                'is_valid': False
            })
    
    return run_results


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


def main():
    parser = argparse.ArgumentParser(description='Classic Tabu Search for Maximum Flow Problem')
    parser.add_argument('--network', type=str, required=True, 
                       help='Network file to process')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Configuration file path')
    parser.add_argument('--parallel', action='store_true',
                       help='Use parallel execution')
    args = parser.parse_args()
    
    try:
        config = load_config(args.config)
        print(f"Configuration loaded")
        print(f"Runs per experiment: {config['experiments']['num_runs']}")
        
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return
    
    if not os.path.exists(args.network):
        print(f"Error: Network file {args.network} not found")
        return
    
    print(f"Processing network: {args.network}")
    
    try:
        network_data = read_network(args.network, verbose=True)
    except Exception as e:
        print(f"Error loading network: {e}")
        return
    
    # Choose execution mode
    if args.parallel:
        run_results = run_parallel_experiments(network_data, config)
    else:
        run_results = run_sequential_experiments(network_data, config)
    
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

    network_number = os.path.basename(args.network).replace('network_', '').replace('.txt', '')
    output_dir = f"data/results/network_{network_number}"
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'network_file': args.network,
        'network_info': network_data.info,
        'run_results': run_results,
        'aggregate_stats': aggregate_stats,
        'theoretical_max': theoretical_max,
        'parallel_execution': args.parallel
    }
    
    if config['output']['save_results']:
        clean_results = results.copy()
        for run_result in clean_results.get('run_results', []):
            if 'final_flow' in run_result:
                del run_result['final_flow']
        
        filename = f"results_network_{network_number}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(clean_results, f, indent=2)
        
        print(f"Results saved to {filepath}")
    
    print(f"\nRESULTS SUMMARY")
    print("-" * 40)
    print(f"Best: {aggregate_stats['best']:.4f}")
    print(f"Mean: {aggregate_stats['mean']:.4f}")
    print(f"StdDev: {aggregate_stats['std_dev']:.4f}")
    print(f"Optimal: {aggregate_stats['optimal_runs']}/{len(valid_results)}")
    print(f"AvgTime: {aggregate_stats['mean_execution_time']:.1f}s")
    print(f"AvgIter: {aggregate_stats['mean_iterations_to_best']:.0f}")
    print("-" * 40)
    
    create_plots(results, network_number, config)


if __name__ == "__main__":
    main()