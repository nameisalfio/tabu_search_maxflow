import random
import copy
import networkx as nx
import yaml
from typing import Dict, List, Tuple, Optional


class TabuMove:
    """
    Representation of a move in the Tabu Search.
    A move can modify multiple arcs to maintain flow conservation.
    """
    
    def __init__(self, modified_arcs: List[Tuple[int, int]], delta_values: List[float], move_type: str):
        self.modified_arcs = modified_arcs  # List of arcs modified
        self.delta_values = delta_values    # Corresponding flow changes
        self.move_type = move_type          # Description of move type
    
    def get_reverse(self) -> 'TabuMove':
        """
        Get the reverse move.
        ========================
        If I increase flow on a path by +1, the reverse move is 
        decreasing flow on the same path by -1.
        
        The tabu list stores recent moves to prevent immediate backtracking.
        Without this mechanism, the algorithm could get stuck in loops!
        """
        reverse_deltas = [-delta for delta in self.delta_values]
        reverse_type = f"reverse_{self.move_type}"
        return TabuMove(self.modified_arcs.copy(), reverse_deltas, reverse_type)
    
    def __str__(self):
        return f"{self.move_type} on {len(self.modified_arcs)} arcs"


class TabuSearchMaxFlow:
    """
    Tabu Search algorithm for Maximum Flow Problem.
    
    Memory Structure:
    - Short-term memory: List of recent moves (tabu_list) 
    - Medium-term memory: Counter of arc modifications (arc_frequency)
    - Long-term memory: List of best solutions found (elite_solutions)
    """
    
    def __init__(self, graph: nx.DiGraph, source: int, sink: int, config_path: str = "config.yaml"):
        self.graph = graph
        self.source = source
        self.sink = sink
        
        # Load configuration
        self.config = self._load_config(config_path)
        self._initialize_parameters()

        # Initialize edges and capacities early
        self.edges = list(self.graph.edges())
        self.edge_capacities = {(u, v): self.graph[u][v]['capacity'] for u, v in self.edges}
        self.function_evaluations = 0

        # Algorithm state
        self.current_solution = self._get_initial_solution()
        self.best_solution = copy.deepcopy(self.current_solution)
        self.best_value = self._evaluate_solution(self.best_solution)

        self._initialize_memory_structures()
                
        # Statistics
        self.iteration = 0
        self.convergence_history = []
        self.no_improvement_count = 0
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            raise Exception(f"Error loading config from {config_path}: {e}")
    
    def _initialize_parameters(self):
        tabu_config = self.config['tabu_search']
        self.max_iterations = tabu_config['max_iterations']
        self.tabu_list_size = tabu_config['tabu_list_size']
        self.aspiration_enabled = tabu_config.get('aspiration_enabled', True)
        self.delta_step = tabu_config.get('delta_step', 1.0)
        self.intensification_threshold = tabu_config.get('intensification_threshold', 50)
        self.diversification_threshold = tabu_config.get('diversification_threshold', 100)
        self.elite_size = tabu_config.get('elite_size', 5)
    
        seeds = self.config.get('experiments', {}).get('random_seeds', [42])
        random.seed(seeds[0])
    
    def _initialize_memory_structures(self):
        """Initialize memory structures."""
        self.tabu_list = []  # List of TabuMove objects
        self.arc_frequency = {edge: 0 for edge in self.edges}
        self.elite_solutions = []
    
    def _get_initial_solution(self) -> Dict[Tuple[int, int], float]:
        """Get initial solution s0 (empty flow)."""
        return {edge: 0.0 for edge in self.edges}
    
    def _evaluate_solution(self, solution: Dict[Tuple[int, int], float]) -> float:
        """Evaluate solution quality (total flow from source)."""
        self.function_evaluations += 1
        return sum(solution.get((self.source, v), 0.0) for v in self.graph.successors(self.source))
    
    def _is_feasible(self, solution: Dict[Tuple[int, int], float]) -> bool:
        """Check if solution satisfies all flow constraints."""
        
        # Check capacity constraints: 0 ≤ flow ≤ capacity
        for (u, v), flow in solution.items():
            if flow < -1e-9 or flow > self.edge_capacities[(u, v)] + 1e-9:
                return False
        
        # Check flow conservation: flow_in = flow_out for intermediate nodes
        for node in self.graph.nodes():
            if node == self.source or node == self.sink:
                continue  # Source and sink don't need flow conservation
                
            flow_in = sum(solution.get((u, node), 0.0) for u in self.graph.predecessors(node))
            flow_out = sum(solution.get((node, v), 0.0) for v in self.graph.successors(node))
            
            if abs(flow_in - flow_out) > 1e-9:
                return False
        
        return True
    
    def _find_augmenting_path(self, current_solution: Dict[Tuple[int, int], float]) -> Optional[List[Tuple[int, int]]]:
        """
        Find an augmenting path from source to sink in the residual graph.
        """
        # Create residual graph
        residual_graph = nx.DiGraph()
        
        for (u, v), capacity in self.edge_capacities.items():
            current_flow = current_solution.get((u, v), 0.0)
            residual_capacity = capacity - current_flow
            
            # Add forward edge if there's remaining capacity
            if residual_capacity > 1e-9:
                residual_graph.add_edge(u, v, capacity=residual_capacity)
            
            # Add backward edge if there's current flow (can be reduced)
            if current_flow > 1e-9:
                residual_graph.add_edge(v, u, capacity=current_flow)
        
        # Find shortest path in residual graph
        try:
            node_path = nx.shortest_path(residual_graph, self.source, self.sink)
            # Convert node path to edge path
            edge_path = [(node_path[i], node_path[i+1]) for i in range(len(node_path)-1)]
            return edge_path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
    
    def _generate_path_moves(self, current_solution: Dict[Tuple[int, int], float]) -> List[Tuple[TabuMove, Dict[Tuple[int, int], float]]]:
        """
        Generate moves based on augmenting paths.
        This ensures flow conservation is maintained.
        """
        neighbors = []
        
        # Try to find multiple augmenting paths
        for attempt in range(10):  # Try up to 10 different paths
            path = self._find_augmenting_path(current_solution)
            if not path:
                break
            
            # Calculate bottleneck capacity on this path
            bottleneck = float('inf')
            for edge in path:
                if edge in self.edge_capacities:
                    # Forward edge
                    current_flow = current_solution.get(edge, 0.0)
                    capacity = self.edge_capacities[edge]
                    bottleneck = min(bottleneck, capacity - current_flow)
                else:
                    # Backward edge (reduce existing flow)
                    reverse_edge = (edge[1], edge[0])
                    if reverse_edge in current_solution:
                        current_flow = current_solution[reverse_edge]
                        bottleneck = min(bottleneck, current_flow)
            
            if bottleneck > 1e-9:
                # Create move with limited flow increase
                flow_increase = min(bottleneck, self.delta_step)
                
                move_arcs = []
                move_deltas = []
                
                for edge in path:
                    if edge in self.edge_capacities:
                        # Forward edge - increase flow
                        move_arcs.append(edge)
                        move_deltas.append(flow_increase)
                    else:
                        # Backward edge - decrease flow on reverse edge
                        reverse_edge = (edge[1], edge[0])
                        move_arcs.append(reverse_edge)
                        move_deltas.append(-flow_increase)
                
                move = TabuMove(move_arcs, move_deltas, f"augmenting_path_{attempt}")
                new_solution = self._apply_move(current_solution, move)
                
                if new_solution and self._is_feasible(new_solution):
                    neighbors.append((move, new_solution))
                
                # Modify current solution temporarily to find different paths
                temp_solution = copy.deepcopy(current_solution)
                for arc, delta in zip(move_arcs, move_deltas):
                    if arc in temp_solution:
                        temp_solution[arc] += delta
                current_solution = temp_solution
        
        return neighbors
    
    def _generate_single_arc_moves(self, current_solution: Dict[Tuple[int, int], float]) -> List[Tuple[TabuMove, Dict[Tuple[int, int], float]]]:
        """
        Generate simple single-arc moves.
        Only works when flow conservation can be maintained.
        """
        neighbors = []
        
        for edge in self.edges:
            current_flow = current_solution.get(edge, 0.0)
            capacity = self.edge_capacities[edge]
            
            # Try increasing flow (only on source edges or when flow conservation allows)
            u, v = edge
            if u == self.source or v == self.sink:  # Source out or sink in edges
                if current_flow + self.delta_step <= capacity:
                    move = TabuMove([edge], [self.delta_step], "increase_terminal")
                    new_solution = self._apply_move(current_solution, move)
                    if new_solution and self._is_feasible(new_solution):
                        neighbors.append((move, new_solution))
            
            # Try decreasing flow
            if current_flow - self.delta_step >= 0:
                move = TabuMove([edge], [-self.delta_step], "decrease_single")
                new_solution = self._apply_move(current_solution, move)
                if new_solution and self._is_feasible(new_solution):
                    neighbors.append((move, new_solution))
        
        return neighbors
    
    def _generate_neighborhood(self, current_solution: Dict[Tuple[int, int], float]) -> List[Tuple[TabuMove, Dict[Tuple[int, int], float]]]:
        """
        Generate neighborhood N(s): all possible valid moves from current solution.
        Combines path-based moves with single-arc moves.
        """
        neighbors = []
        
        # Strategy 1: Path-based moves (most important for flow problems)
        path_neighbors = self._generate_path_moves(current_solution)
        neighbors.extend(path_neighbors)
        
        # Strategy 2: Single arc moves (for fine-tuning)
        single_neighbors = self._generate_single_arc_moves(current_solution)
        neighbors.extend(single_neighbors)
        
        print(f"    Generated {len(neighbors)} neighbors")
        return neighbors
    
    def _apply_move(self, solution: Dict[Tuple[int, int], float], move: TabuMove) -> Optional[Dict[Tuple[int, int], float]]:
        """Apply a move to get new solution."""
        new_solution = copy.deepcopy(solution)
        
        # Apply all arc modifications
        for edge, delta in zip(move.modified_arcs, move.delta_values):
            if edge in new_solution:
                new_solution[edge] += delta
            else:
                new_solution[edge] = delta
                
            # Ensure non-negative flows
            new_solution[edge] = max(0.0, new_solution[edge])
        
        return new_solution
    
    def _is_move_tabu(self, move: TabuMove) -> bool:
        """Check if move is tabu."""
        reverse_move = move.get_reverse()
        
        for tabu_move in self.tabu_list:
            # Check if moves affect same arcs with similar changes
            if (set(tabu_move.modified_arcs) == set(reverse_move.modified_arcs) and
                len(tabu_move.delta_values) == len(reverse_move.delta_values)):
                
                # Check if deltas are approximately reverse
                all_reverse = True
                for t_delta, r_delta in zip(tabu_move.delta_values, reverse_move.delta_values):
                    if abs(t_delta - r_delta) > 1e-9:
                        all_reverse = False
                        break
                
                if all_reverse:
                    return True
        
        return False
    
    def _aspiration_criterion(self, move: TabuMove, solution_value: float) -> bool:
        """
        Check aspiration criterion: better than best known solution found so far.
        This allows tabu moves if they improve upon our current best.
        """
        return self.aspiration_enabled and solution_value > self.best_value
    
    def _update_memory_structures(self, move: TabuMove, solution: Dict[Tuple[int, int], float], solution_value: float):
        """Update all memory structures."""
        self.tabu_list.append(move)
        
        if len(self.tabu_list) > self.tabu_list_size:
            self.tabu_list.pop(0)  
        
        # Update frequency for all modified arcs
        for edge in move.modified_arcs:
            if edge in self.arc_frequency:
                self.arc_frequency[edge] += 1
        
        self._update_elite_solutions(solution, solution_value)
    
    def _update_elite_solutions(self, solution: Dict[Tuple[int, int], float], value: float):
        """Update elite solutions list."""
        self.elite_solutions.append((copy.deepcopy(solution), value))
        self.elite_solutions.sort(key=lambda x: x[1], reverse=True)
        if len(self.elite_solutions) > self.elite_size:
            self.elite_solutions = self.elite_solutions[:self.elite_size]
    
    def _intensification_criterion(self) -> bool:
        return self.no_improvement_count >= self.intensification_threshold
    
    def _diversification_criterion(self) -> bool:
        return self.no_improvement_count >= self.diversification_threshold
    
    def _apply_intensification(self):
        print(f"Applying intensification")
        self.no_improvement_count = max(0, self.no_improvement_count // 2)
    
    def _apply_diversification(self):
        print(f"Applying diversification")
        
        if self.elite_solutions:
            elite_solution, _ = random.choice(self.elite_solutions)
            self.current_solution = copy.deepcopy(elite_solution)
        
        self.tabu_list.clear()
        self.no_improvement_count = 0
    
    def _stopping_criteria_satisfied(self) -> bool:
        return self.iteration >= self.max_iterations
    
    def solve(self) -> Tuple[Dict[Tuple[int, int], float], float, List[float]]:
        """Main Tabu Search algorithm."""
        print(f"Starting Tabu Search")
        print('-' * 70)
        print(f"Max iterations: {self.max_iterations}, Tabu size: {self.tabu_list_size}")
        
        while not self._stopping_criteria_satisfied():
            self.iteration += 1
            
            neighbors = self._generate_neighborhood(self.current_solution)
            
            if not neighbors:
                print(f"No neighbors at iteration {self.iteration}, stopping")
                break
            
            best_move = None
            best_neighbor = None
            best_neighbor_value = float('-inf')
            
            # Find best admissible move
            for move, neighbor in neighbors:
                neighbor_value = self._evaluate_solution(neighbor)
                
                if (not self._is_move_tabu(move) or 
                    self._aspiration_criterion(move, neighbor_value)):
                    
                    if neighbor_value > best_neighbor_value:
                        best_move = move
                        best_neighbor = neighbor
                        best_neighbor_value = neighbor_value
            
            # If no admissible move, take least bad tabu move
            if best_move is None:
                for move, neighbor in neighbors:
                    neighbor_value = self._evaluate_solution(neighbor)
                    if neighbor_value > best_neighbor_value:
                        best_move = move
                        best_neighbor = neighbor
                        best_neighbor_value = neighbor_value
            
            # Move to best neighbor
            if best_move is not None:
                self.current_solution = best_neighbor
                current_value = best_neighbor_value
                
                # Update best solution if improved
                if current_value > self.best_value:
                    self.best_solution = copy.deepcopy(self.current_solution)
                    self.best_value = current_value
                    self.no_improvement_count = 0
                    print(f"Iteration {self.iteration}: new best = {self.best_value:.2f}")
                else:
                    self.no_improvement_count += 1
                
                self._update_memory_structures(best_move, self.current_solution, current_value)
                self.convergence_history.append(self.best_value)
                
                if self._intensification_criterion():
                    self._apply_intensification()
                
                if self._diversification_criterion():
                    self._apply_diversification()
                
                if self.iteration % 500 == 0:
                    print(f"Iteration {self.iteration}: current = {current_value:.2f}, best = {self.best_value:.2f}")
            
            else:
                self.convergence_history.append(self.best_value)
        
        print(f"Completed: {self.iteration} iterations, best = {self.best_value:.2f}")
        
        return self.best_solution, self.best_value, self.convergence_history
    
    def get_function_evaluations(self) -> int:
        return self.function_evaluations
    
    def get_memory_statistics(self) -> dict:
        return {
            'tabu_list_size': len(self.tabu_list),
            'current_tabu_moves': [str(move) for move in self.tabu_list],
            'most_frequent_arcs': sorted(self.arc_frequency.items(), key=lambda x: x[1], reverse=True)[:5],
            'elite_solutions_count': len(self.elite_solutions),
            'elite_values': [value for _, value in self.elite_solutions]
        }