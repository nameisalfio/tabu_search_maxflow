# src/algorithms/tabu_search.py
import logging
import random
from collections import deque
from typing import Dict, Tuple, List, Optional
import networkx as nx
from src.data.network_reader import NetworkData

class TabuSearch:
    # *** CORREZIONE 1: Aggiunto 'show_logs: bool = True' alla firma del costruttore ***
    def __init__(self, network_data: NetworkData, config: dict, seed: int, run_id: int, logger: logging.Logger, show_logs: bool = True):
        self.network = network_data
        self.config = config
        self.graph = network_data.graph
        self.source = network_data.source
        self.sink = network_data.sink
        self.random = random.Random(seed)
        self.tabu_list = deque(maxlen=config['tabu_search']['tabu_list_size'])
        self.run_id = run_id
        self.logger = logger
        # *** CORREZIONE 2: Salvataggio del valore del nuovo argomento ***
        self.show_logs = show_logs
        self.current_flow_dict = {edge: 0.0 for edge in self.graph.edges}
        self.current_flow_value = 0.0
        self.best_flow_value = 0.0
        self.iteration_of_best = 0
        self.evaluations_to_best = 0
        self.history = []
        
    def _find_augmenting_path(self) -> Optional[Tuple[List[int], float]]:
        # Questo metodo rimane invariato
        queue = deque([(self.source, [self.source], float('inf'))])
        visited = {self.source}
        paths = []
        while queue:
            u, path, path_flow = queue.popleft()
            successors = list(self.graph.successors(u))
            self.random.shuffle(successors)
            for v in successors:
                residual_capacity = self.graph[u][v]['capacity'] - self.current_flow_dict.get((u, v), 0)
                if v not in visited and residual_capacity > 1e-9:
                    new_path_flow = min(path_flow, residual_capacity)
                    new_path = path + [v]
                    if v == self.sink:
                        paths.append((new_path, new_path_flow))
                    else:
                        visited.add(v)
                        queue.append((v, new_path, new_path_flow))
        if paths:
            return max(paths, key=lambda item: item[1])
        return None

    def _get_bottleneck_edge(self, path: List[int], bottleneck_capacity: float) -> Tuple[int, int]:
        # Questo metodo rimane invariato
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            if self.graph.has_edge(u, v):
                residual = self.graph[u][v]['capacity'] - self.current_flow_dict.get((u,v), 0)
                if abs(residual - bottleneck_capacity) < 1e-9:
                    return u, v
        return path[0], path[1]

    def run(self):
        """Main loop for Tabu Search. I log sono ora condizionali."""
        
        # *** CORREZIONE 3: Tutti i log sono ora condizionati da 'self.show_logs' ***
        if self.show_logs:
            self.logger.info(f"    --- Starting Run {self.run_id:02d} (Seed: {self.random.getstate()[1][0]}) ---")
        
        max_iterations = self.config['tabu_search']['max_iterations']
        optimal_flow_bound = self.network.get_max_possible_flow()
        num_evaluations = 0

        for i in range(max_iterations):
            move = self._find_augmenting_path()
            num_evaluations += 1

            if move is None:
                if self.show_logs:
                    self.logger.info(f"      [Run {self.run_id:02d}][Iter {i:05d}] No more augmenting paths found. Terminating.")
                break

            path, flow_to_push = move
            bottleneck_edge = self._get_bottleneck_edge(path, flow_to_push)
            reverse_bottleneck = (bottleneck_edge[1], bottleneck_edge[0])
            is_tabu = reverse_bottleneck in self.tabu_list
            new_potential_flow = self.current_flow_value + flow_to_push
            if new_potential_flow > self.best_flow_value: 
                is_tabu = False
            
            if not is_tabu:
                self.current_flow_value += flow_to_push
                for j in range(len(path) - 1):
                    u, v = path[j], path[j+1]
                    self.current_flow_dict[(u, v)] += flow_to_push
                self.tabu_list.append(bottleneck_edge)
                
                if self.current_flow_value > self.best_flow_value:
                    self.best_flow_value = self.current_flow_value
                    self.iteration_of_best = i
                    self.evaluations_to_best = num_evaluations
                    if self.show_logs:
                        self.logger.info(f"      [Run {self.run_id:02d}][Iter {i:05d}] New best flow: {self.best_flow_value:.6f}")
            
            self.history.append(self.best_flow_value)
            
            if abs(self.best_flow_value - optimal_flow_bound) < 1e-3:
                if self.show_logs:
                    self.logger.info(f"      [Run {self.run_id:02d}][Iter {i:05d}] Optimal flow reached. Terminating early.")
                break
        
        if self.show_logs:
            self.logger.info(f"    --- Finished Run {self.run_id:02d}. Final Best Flow: {self.best_flow_value:.6f} ---\n")
        
        return {
            "run_id": self.run_id, "best_flow": self.best_flow_value,
            "iteration_of_best": self.iteration_of_best,
            "evaluations_to_best": self.evaluations_to_best,
            "history": self.history
        }