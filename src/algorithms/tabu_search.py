import logging
import random
import copy
from collections import deque, defaultdict
from typing import Tuple, List, Optional, Dict, Set
from src.data.network_reader import NetworkData

class TabuSearch:
    
    def __init__(self, network_data: NetworkData, config: dict, seed: int, run_id: int, 
                 logger: logging.Logger, show_logs: bool = True, progress_dict=None):
        
        # Core components
        self.network = network_data
        self.config = config
        self.graph = network_data.graph
        self.source = network_data.source
        self.sink = network_data.sink
        self.random = random.Random(seed)
        self.run_id = run_id
        self.logger = logger
        self.show_logs = show_logs
        self.progress_dict = progress_dict
        
        # Current solution state
        self.current_flow_dict = {edge: 0.0 for edge in self.graph.edges}
        self.current_flow_value = 0.0
        
        # Best global solution found
        self.best_flow_value = 0.0
        self.best_flow_dict = {}
        
        # Performance tracking
        self.iteration_of_best = 0
        self.evaluations_to_best = 0
        self.history = []

        self.strategies_config = config.get('strategies', {})
        self.stagnation_counter = 0
        self.tabu_list = deque(maxlen=config['tabu_search']['tabu_list_size'])
        self.elite_solutions = []
        self.intensify_config = self.strategies_config.get('intensification', {})
        self.diversify_config = self.strategies_config.get('diversification', {})
            
        # 1. Long-term frequency memory
        self.frequency_config = config.get('frequency', {})
        self.frequency_memory = defaultdict(int)
        self.alpha = self.frequency_config.get('alpha', 0.01)
        self.enable_frequency_memory = self.frequency_config.get('enabled', True)
        
        self.frequency_penalties_applied = 0
        self.total_moves_evaluated = 0
        
        if self.show_logs:
            self.logger.info(f"      [Run {self.run_id:02d}] Enhanced Tabu Search initialized")
            self.logger.info(f"      [Run {self.run_id:02d}] Frequency memory: {self.enable_frequency_memory} (alpha={self.alpha})")

    def _find_augmenting_path(self) -> Optional[Tuple[List[int], float]]:
        """Find augmenting path from source to sink using BFS"""
        queue = deque([(self.source, [self.source], float('inf'))])
        visited = {self.source}

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
                        return (new_path, new_path_flow)
                    
                    visited.add(v)
                    queue.append((v, new_path, new_path_flow))
        return None

    def _get_bottleneck_edge(self, path: List[int], bottleneck_capacity: float) -> Tuple[int, int]:
        """Identify the bottleneck edge in the path"""
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            if self.graph.has_edge(u, v):
                residual = self.graph[u][v]['capacity'] - self.current_flow_dict.get((u,v), 0)
                if abs(residual - bottleneck_capacity) < 1e-9:
                    return u, v
        return path[0], path[1]

    def _update_elite_solutions(self, flow_value: float, flow_dict: Dict[Tuple[int, int], float]):
        """Update elite solutions list"""
        if not self.intensify_config.get('elite_solutions_count', 0) > 0: 
            return
            
        self.elite_solutions.append((flow_value, copy.deepcopy(flow_dict)))
        self.elite_solutions.sort(key=lambda x: x[0], reverse=True)
        max_elite = self.intensify_config['elite_solutions_count']
        self.elite_solutions = self.elite_solutions[:max_elite]

    def _perform_intensification(self):
        """Perform intensification by restarting from best elite solution"""
        if not self.elite_solutions: 
            return
            
        if self.show_logs: 
            self.logger.info(f"      [Run {self.run_id:02d}] --- INTENSIFICATION --- Restarting from elite solution")
            
        elite_value, elite_dict = self.elite_solutions[0]
        self.current_flow_value = elite_value
        self.current_flow_dict = copy.deepcopy(elite_dict)
        self.tabu_list.clear()

    def _perform_diversification(self):
        """Perform diversification by resetting the flow"""
        if self.show_logs: 
            self.logger.info(f"      [Run {self.run_id:02d}] --- DIVERSIFICATION --- Resetting flow")
            
        self.current_flow_dict = {edge: 0.0 for edge in self.graph.edges}
        self.current_flow_value = 0.0
        self.tabu_list.clear()
    
    def _update_frequency_memory(self, move: Tuple[int, int]):
        """Update frequency memory for the given move"""
        if self.enable_frequency_memory:
            self.frequency_memory[move] += 1
    
    def _get_frequency_penalty(self, move: Tuple[int, int]) -> float:
        """Calculate frequency penalty for a move"""
        if not self.enable_frequency_memory:
            return 0.0
        
        frequency = self.frequency_memory.get(move, 0)
        penalty = self.alpha * frequency
        
        if penalty > 0:
            self.frequency_penalties_applied += 1
            
        return penalty
    
    def _evaluate_move_with_frequency(self, move: Tuple[List[int], float]) -> float:
        """Evaluate a move considering frequency penalty"""
        path, flow_to_push = move
        bottleneck_edge = self._get_bottleneck_edge(path, flow_to_push)

        base_value = flow_to_push
        frequency_penalty = self._get_frequency_penalty(bottleneck_edge)
        final_value = base_value - frequency_penalty
        
        self.total_moves_evaluated += 1
        
        return final_value

    
    def _solution_to_signature(self, flow_dict: Dict[Tuple[int, int], float]) -> frozenset:
        """Convert solution to signature for comparison"""
        active_edges = frozenset((u, v) for (u, v), flow in flow_dict.items() if flow > 1e-9)
        return active_edges
    
    def _calculate_solution_distance(self, source_dict: Dict[Tuple[int, int], float], 
                                     target_dict: Dict[Tuple[int, int], float]) -> int:
        """Calculate Hamming distance between two solutions"""
        source_sig = self._solution_to_signature(source_dict)
        target_sig = self._solution_to_signature(target_dict)
        
        distance = len(source_sig.symmetric_difference(target_sig))
        return distance
    
    def _find_move_towards_target(self, source_dict: Dict[Tuple[int, int], float], 
                                  target_dict: Dict[Tuple[int, int], float]) -> Optional[Tuple[int, int]]:
        """Find a move that makes source more similar to target"""
        source_sig = self._solution_to_signature(source_dict)
        target_sig = self._solution_to_signature(target_dict)
        
        edges_to_activate = target_sig - source_sig
        edges_to_deactivate = source_sig - target_sig
        all_moves = list(edges_to_activate) + list(edges_to_deactivate)
        
        if not all_moves:
            return None
            
        selected_move = self.random.choice(all_moves)
        return selected_move
    
    def run(self):
        """Main enhanced tabu search algorithm"""
        if self.show_logs: 
            self.logger.info(f"    --- Starting Enhanced Run {self.run_id:02d} (Seed: {self.random.getstate()[1][0]}) ---")
        
        max_iterations = self.config['tabu_search']['max_iterations']
        optimal_flow_bound = self.network.get_max_possible_flow()
        num_evaluations = 0
        intensify_limit = self.intensify_config.get('stagnation_limit', 0)
        diversify_limit = self.diversify_config.get('reset_limit', 0)

        for i in range(max_iterations):
            if diversify_limit > 0 and self.stagnation_counter >= diversify_limit:
                self._perform_diversification()
                self.stagnation_counter = 0
            elif intensify_limit > 0 and self.stagnation_counter >= intensify_limit:
                self._perform_intensification()
                self.stagnation_counter = 0

            if self.progress_dict is not None:
                self.progress_dict[self.run_id] = {
                    'iter': i, 
                    'max_iter': max_iterations, 
                    'flow': self.best_flow_value,
                    'freq_penalties': self.frequency_penalties_applied,
                    'moves_evaluated': self.total_moves_evaluated
                }

            move = self._find_augmenting_path()
            num_evaluations += 1

            if move is None:
                self.stagnation_counter = max_iterations
                continue

            path, flow_to_push = move
            bottleneck_edge = self._get_bottleneck_edge(path, flow_to_push)
            
            is_tabu = bottleneck_edge in self.tabu_list
            new_potential_flow = self.current_flow_value + flow_to_push
            aspiration_holds = new_potential_flow > self.best_flow_value
            
            move_value = self._evaluate_move_with_frequency(move)
            
            if not is_tabu or aspiration_holds:
                self.current_flow_value += flow_to_push
                for j in range(len(path) - 1):
                    u, v = path[j], path[j+1]
                    self.current_flow_dict[(u, v)] += flow_to_push
                
                self.tabu_list.append(bottleneck_edge)
                self._update_frequency_memory(bottleneck_edge)
                
                if self.current_flow_value > self.best_flow_value:
                    self.best_flow_value = self.current_flow_value
                    self.best_flow_dict = copy.deepcopy(self.current_flow_dict)
                    self.iteration_of_best = i
                    self.evaluations_to_best = num_evaluations
                    self.stagnation_counter = 0
                    self._update_elite_solutions(self.best_flow_value, self.best_flow_dict)
                    
                    if self.show_logs: 
                        self.logger.info(f"      [Run {self.run_id:02d}][Iter {i:05d}] New best flow: {self.best_flow_value:.6f}")
                else:
                    self.stagnation_counter += 1

            self.history.append(self.best_flow_value)
            
            if abs(self.best_flow_value - optimal_flow_bound) < 1e-3:
                if self.show_logs: 
                    self.logger.info(f"      [Run {self.run_id:02d}][Iter {i:05d}] Optimal flow reached. Terminating early.")
                break
        
        if self.show_logs: 
            self.logger.info(f"    --- Finished Enhanced Run {self.run_id:02d}. Final Best Flow: {self.best_flow_value:.6f} ---")
            self.logger.info(f"    --- Enhanced Statistics ---")
            self.logger.info(f"    --- Frequency penalties applied: {self.frequency_penalties_applied} ---")
            self.logger.info(f"    --- Total moves evaluated: {self.total_moves_evaluated} ---\n")
        
        return {
            "run_id": self.run_id, 
            "best_flow": self.best_flow_value, 
            "iteration_of_best": self.iteration_of_best, 
            "evaluations_to_best": self.evaluations_to_best, 
            "history": self.history,
            "frequency_penalties_applied": self.frequency_penalties_applied,
            "total_moves_evaluated": self.total_moves_evaluated,
            "frequency_memory_size": len(self.frequency_memory),
            "elite_solutions_count": len(self.elite_solutions)
        }