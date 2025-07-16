import os
import re
import time
import yaml
import argparse
import numpy as np
import pandas as pd
import multiprocessing
from functools import partial
import logging

from src.data.network_reader import NetworkData, read_network, compute_reference_max_flow
from src.algorithms.tabu_search import TabuSearch
from src.visualization.plotter import plot_all_runs_convergence, plot_best_run_convergence
from src.utils.logging_utils import setup_queue_logging, setup_main_logger

def run_single_trial(run_id: int, log_queue, network_data: 'NetworkData', config: dict, show_logs: bool) -> dict:
    """
    Function to execute in each parallel process.
    """
    setup_queue_logging(log_queue)
    logger = logging.getLogger()
    
    seed = config['experiments']['random_seeds'][run_id - 1]
    
    start_time = time.time()
    ts_solver = TabuSearch(network_data, config, seed, run_id, logger, show_logs)
    result = ts_solver.run()
    end_time = time.time()
    
    result['execution_time'] = end_time - start_time
    return result

def process_results(run_results: list, total_execution_time: float, network_data: 'NetworkData', config: dict):
    """
    Takes the collected results and generates summary tables and final plots.
    """
    instance_name = network_data.info['filename'].replace('.txt', '')
    output_dir = os.path.join(config['output']['results_directory'], instance_name)
    reference_flow = compute_reference_max_flow(network_data)
    run_stats = []
    for res in sorted(run_results, key=lambda x: x['run_id']):
        run_stats.append({
            "Run": res["run_id"], "Best Flow": res["best_flow"],
            "Iterations to Best": res["iteration_of_best"], "Evaluations to Best": res["evaluations_to_best"],
            "Execution Time (s)": res["execution_time"], "Is Optimal": abs(res["best_flow"] - reference_flow) < 1e-3
        })
    run_stats_df = pd.DataFrame(run_stats)
    best_flows = run_stats_df["Best Flow"].values
    num_optimal = run_stats_df["Is Optimal"].sum()
    num_runs = len(run_results)
    summary_stats = {
        "best_flow_found": np.max(best_flows), "mean_flow": np.mean(best_flows),
        "std_dev_flow": np.std(best_flows), "mean_iterations_to_best": run_stats_df["Iterations to Best"].mean(),
        "mean_evaluations_to_best": run_stats_df["Evaluations to Best"].mean(),
        "success_rate_perc": (num_optimal / num_runs) * 100,
        "mean_execution_time": run_stats_df["Execution Time (s)"].mean(),
        "total_execution_time": total_execution_time
    }

    summary_log = "="*80 + "\n--- Performance Summary ---\n" + "="*80 + "\n"
    summary_log += f"  1. Best flow found (across all runs):   {summary_stats['best_flow_found']:.6f}\n"
    summary_log += f"  2. Mean of best flows found:             {summary_stats['mean_flow']:.6f}\n"
    summary_log += f"  3. Standard deviation of best flows:     {summary_stats['std_dev_flow']:.6f}\n"
    summary_log += f"  4. Mean iterations to find best solution:{summary_stats['mean_iterations_to_best']:.2f}\n"
    summary_log += f"  5. Mean evaluations to find best solution:{summary_stats['mean_evaluations_to_best']:.2f}\n"
    summary_log += f"  6. Success Rate (reaching optimum):      {summary_stats['success_rate_perc']:.1f}% ({num_optimal}/{num_runs})\n"
    summary_log += f"  Mean execution time per run:             {summary_stats['mean_execution_time']:.4f} seconds\n"
    summary_log += f"  Total execution time for {num_runs} runs:    {summary_stats['total_execution_time']:.4f} seconds\n"
    summary_log += "\n" + "="*80 + "\n--- Detailed Run-by-Run Statistics ---\n" + "="*80 + "\n"
    summary_log += run_stats_df.to_string(index=False)
    
    logging.info(summary_log)
    
    if config['visualization']['save_plots']:
        plot_style = config['visualization'].get('style', 'light')
        logging.info("\n" + "="*80 + f"\n--- Generating Plots (Style: {plot_style}) ---\n" + "="*80)
        all_histories = [res['history'] for res in sorted(run_results, key=lambda x: x['run_id'])]
        best_run_index = np.argmax(best_flows)
        best_history = run_results[best_run_index]['history']
        p1_path = plot_all_runs_convergence(all_histories, instance_name, output_dir, style=plot_style, summary_stats=summary_stats)
        logging.info(f"  -> Saved all runs convergence plot to: {p1_path}")
        p2_path = plot_best_run_convergence(best_history, instance_name, output_dir, style=plot_style)
        logging.info(f"  -> Saved best run convergence plot to: {p2_path}")

def run_experiment_for_instance(network_path: str, config: dict, parallel: bool):
    instance_name = os.path.basename(network_path).replace('.txt', '')
    output_dir = os.path.join(config['output']['results_directory'], instance_name)
    
    with multiprocessing.Manager() as manager:
        log_queue = manager.Queue()

        listener = setup_main_logger(output_dir, instance_name, log_queue)
        listener.start()
        
        main_logger = logging.getLogger()

        main_logger.info("="*60)
        main_logger.info(f"--- STARTING EXPERIMENT FOR INSTANCE: {instance_name} ---")
        show_worker_logs = config['output'].get('show_worker_logs_in_parallel', True)
        if parallel and not show_worker_logs:
             main_logger.info("--- Execution Mode: PARALLEL (Worker logs are hidden) ---")
        else:
             main_logger.info(f"--- Execution Mode: {'PARALLEL' if parallel else 'SEQUENTIAL'} ---")
        main_logger.info("="*60)

        network_data = read_network(network_path)
        main_logger.info(
        f'''\n    ----- Network Info -----
        Filename: {network_data.info["filename"]}
        Number of nodes: {network_data.info["num_nodes"]}
        Number of edges: {network_data.info["num_edges"]}
        Source: {network_data.info["source"]}
        Sink: {network_data.info["sink"]}
        Total source capacity: {network_data.info["total_source_capacity"]}
        Total sink capacity: {network_data.info["total_sink_capacity"]}
    ------------------------\n'''
        )
        
        total_start_time = time.time()
        num_runs = config['experiments']['num_runs']
        
        run_results = []
        
        if parallel:
            worker_func = partial(run_single_trial, 
                                  log_queue=log_queue, 
                                  network_data=network_data, 
                                  config=config,
                                  show_logs=show_worker_logs) 
            with multiprocessing.Pool() as pool:
                run_results = pool.map(worker_func, range(1, num_runs + 1))
        else:
            for i in range(1, num_runs + 1):
                run_id = i
                seed = config['experiments']['random_seeds'][run_id - 1]
                start_time = time.time()
                ts_solver = TabuSearch(network_data, config, seed, run_id, main_logger, show_logs=True)
                result = ts_solver.run()
                end_time = time.time()
                result['execution_time'] = end_time - start_time
                run_results.append(result)

        total_end_time = time.time()
        total_execution_time = total_end_time - total_start_time

        process_results(run_results, total_execution_time, network_data, config)
        
        main_logger.info(f"\n--- Experiment for {instance_name} Complete ---")
        
        listener.stop()

def main():
    parser = argparse.ArgumentParser(description="Run Tabu Search Experiments")
    parser.add_argument('--network', type=str, help='Path to a specific network file.')
    parser.add_argument('--all', action='store_true', help='Run for all networks.')
    parser.add_argument('--parallel', action='store_true', help='Enable parallel execution of runs.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file.')
    args = parser.parse_args()

    with open(args.config, 'r') as f: config = yaml.safe_load(f)

    if args.network:
        run_experiment_for_instance(args.network, config, args.parallel)
    elif args.all:
        network_dir = 'data/networks'
        
        def get_number_from_filename(filename):
            match = re.search(r'(\d+)', filename)
            return int(match.group(1)) if match else 0

        network_files = [f for f in os.listdir(network_dir) if f.endswith(".txt")]
        sorted_files = sorted(network_files, key=get_number_from_filename)
        
        print("\n" + "="*70)
        print("Processing networks in numerical order based on filename:")
        print("="*70)

        for filename in sorted_files:
            network_path = os.path.join(network_dir, filename)
            print(f"\n>>> Processing: {filename}")
            run_experiment_for_instance(network_path, config, args.parallel)
            print("-" * 70)

        print("\nAll experiments are complete.")
    else:
        print("Error: Please specify a network file with --network <path> or use --all.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()