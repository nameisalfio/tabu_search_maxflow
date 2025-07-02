"""
Tabu Search Algorithm for Maximum Flow Problem
"""

import numpy as np
import random
from typing import List, Tuple, Optional
from collections import deque
import time

from ..data.network_reader import NetworkGraph, FlowSolution

class TabuMove:
    """Rappresenta una mossa nel Tabu Search"""
    
    def __init__(self, arc: Tuple[int, int], delta: int, move_type: str):
        self.arc = arc  # (from_node, to_node)
        self.delta = delta
        self.move_type = move_type  # 'increase' or 'decrease'
    
    def __eq__(self, other):
        return (self.arc == other.arc and 
                self.delta == other.delta and 
                self.move_type == other.move_type)
    
    def __hash__(self):
        return hash((self.arc, self.delta, self.move_type))
    
    def __repr__(self):
        return f"Move({self.arc}, {self.delta}, {self.move_type})"

class TabuSearch:
    """Tabu Search per Maximum Flow Problem"""
    
    def __init__(self, max_iterations=20000, tabu_list_size=10, 
                 aspiration_enabled=True, delta_step=1):
        self.max_iterations = max_iterations
        self.tabu_list_size = tabu_list_size
        self.aspiration_enabled = aspiration_enabled
        self.delta_step = delta_step
        
        # Stato interno
        self.tabu_list = deque(maxlen=tabu_list_size)
        self.convergence_history = []
        self.execution_time = 0.0
        self.iterations_executed = 0
    
    def solve(self, graph: NetworkGraph, verbose=True) -> FlowSolution:
        """Risolve il Maximum Flow usando Tabu Search"""
        start_time = time.time()
        
        if verbose:
            print(f"🎯 Starting Tabu Search on {graph}")
            print(f"   Parameters: max_iter={self.max_iterations}, tabu_size={self.tabu_list_size}")
        
        # Soluzione iniziale (greedy)
        current_solution = self._greedy_initial_solution(graph)
        best_solution = current_solution.copy()
        
        self.convergence_history = [current_solution.flow_value]
        
        if verbose:
            print(f"   Initial solution: {current_solution.flow_value}")
        
        # Main loop
        for iteration in range(self.max_iterations):
            self.iterations_executed = iteration + 1
            
            # Genera vicinato
            neighbor_moves = self._generate_neighborhood(graph, current_solution)
            
            if not neighbor_moves:
                if verbose:
                    print(f"   No moves available at iteration {iteration}")
                break
            
            # Seleziona migliore mossa ammissibile
            best_move = self._select_best_move(graph, current_solution, neighbor_moves)
            
            if best_move is None:
                if verbose:
                    print(f"   No admissible moves at iteration {iteration}")
                break
            
            # Applica mossa
            current_solution = self._apply_move(current_solution, best_move)
            self.tabu_list.append(best_move)
            
            # Aggiorna best
            if current_solution.flow_value > best_solution.flow_value:
                best_solution = current_solution.copy()
                if verbose:
                    print(f"   🎉 New best: {best_solution.flow_value} (iter {iteration})")
            
            # Salva convergenza
            self.convergence_history.append(current_solution.flow_value)
            
            # Progress log
            if verbose and iteration > 0 and iteration % 5000 == 0:
                print(f"   Iteration {iteration}: current={current_solution.flow_value}, best={best_solution.flow_value}")
        
        self.execution_time = time.time() - start_time
        
        if verbose:
            print(f"✅ Tabu Search completed!")
            print(f"   Best solution: {best_solution.flow_value}")
            print(f"   Iterations: {self.iterations_executed}")
            print(f"   Execution time: {self.execution_time:.2f}s")
        
        return best_solution
    
    def _greedy_initial_solution(self, graph: NetworkGraph) -> FlowSolution:
        """Genera soluzione iniziale con algoritmo greedy (Ford-Fulkerson semplificato)"""
        flow_matrix = np.zeros((graph.num_nodes, graph.num_nodes))
        total_flow = 0
        
        # Implementazione semplificata di Ford-Fulkerson
        capacity = graph.capacity_matrix.copy()
        
        while True:
            # Trova cammino aumentante con DFS semplice
            path = self._find_augmenting_path(graph, capacity, flow_matrix)
            if not path:
                break
            
            # Trova bottleneck
            bottleneck = float('inf')
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                bottleneck = min(bottleneck, capacity[u][v] - flow_matrix[u][v])
            
            # Aumenta flusso
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                flow_matrix[u][v] += bottleneck
            
            total_flow += bottleneck
        
        return FlowSolution(flow_matrix, total_flow)
    
    def _find_augmenting_path(self, graph: NetworkGraph, capacity: np.ndarray, 
                            flow: np.ndarray) -> List[int]:
        """Trova cammino aumentante con DFS"""
        visited = [False] * graph.num_nodes
        path = []
        
        def dfs(node, target):
            if node == target:
                return True
            
            visited[node] = True
            
            for neighbor in graph.adjacency_list[node]:
                if not visited[neighbor] and (capacity[node][neighbor] - flow[node][neighbor]) > 0:
                    path.append(neighbor)
                    if dfs(neighbor, target):
                        return True
                    path.pop()
            
            return False
        
        path.append(graph.source)
        if dfs(graph.source, graph.sink):
            return path
        return []
    
    def _generate_neighborhood(self, graph: NetworkGraph, solution: FlowSolution) -> List[TabuMove]:
        """Genera mosse del vicinato"""
        moves = []
        
        for u in range(graph.num_nodes):
            for v in range(graph.num_nodes):
                if graph.capacity_matrix[u][v] > 0:  # Arco esistente
                    current_flow = solution.flow_matrix[u][v]
                    capacity = graph.capacity_matrix[u][v]
                    
                    # Mossa incremento
                    if current_flow < capacity:
                        moves.append(TabuMove((u, v), self.delta_step, 'increase'))
                    
                    # Mossa decremento
                    if current_flow > 0:
                        moves.append(TabuMove((u, v), self.delta_step, 'decrease'))
        
        return moves
    
    def _select_best_move(self, graph: NetworkGraph, solution: FlowSolution, 
                         moves: List[TabuMove]) -> Optional[TabuMove]:
        """Seleziona la migliore mossa ammissibile"""
        best_move = None
        best_value = float('-inf')
        
        for move in moves:
            # Check se tabu
            is_tabu = move in self.tabu_list
            
            # Valuta mossa
            temp_value = self._evaluate_move(graph, solution, move)
            
            if temp_value == float('-inf'):  # Mossa non fattibile
                continue
            
            # Applica o non applica in base a tabu e aspirazione
            if not is_tabu:
                if temp_value > best_value:
                    best_move = move
                    best_value = temp_value
            elif self.aspiration_enabled and temp_value > self._get_best_known_value():
                # Criterio di aspirazione
                if temp_value > best_value:
                    best_move = move
                    best_value = temp_value
        
        return best_move
    
    def _evaluate_move(self, graph: NetworkGraph, solution: FlowSolution, move: TabuMove) -> float:
        """Valuta una mossa senza applicarla"""
        temp_flow = solution.flow_matrix.copy()
        u, v = move.arc
        
        if move.move_type == 'increase':
            temp_flow[u][v] += move.delta
        else:
            temp_flow[u][v] -= move.delta
        
        # Check feasibility
        if not self._is_feasible(graph, temp_flow):
            return float('-inf')
        
        # Calcola valore flusso
        return float(np.sum(temp_flow[graph.source, :]))
    
    def _apply_move(self, solution: FlowSolution, move: TabuMove) -> FlowSolution:
        """Applica una mossa e ritorna nuova soluzione"""
        new_flow = solution.flow_matrix.copy()
        u, v = move.arc
        
        if move.move_type == 'increase':
            new_flow[u][v] += move.delta
        else:
            new_flow[u][v] -= move.delta
        
        new_value = float(np.sum(new_flow[graph.source, :]))
        return FlowSolution(new_flow, new_value)
    
    def _is_feasible(self, graph: NetworkGraph, flow_matrix: np.ndarray) -> bool:
        """Controlla fattibilità di un flusso"""
        # Check capacity constraints
        for u in range(graph.num_nodes):
            for v in range(graph.num_nodes):
                if flow_matrix[u][v] < 0 or flow_matrix[u][v] > graph.capacity_matrix[u][v]:
                    return False
        
        # Check flow conservation
        for node in range(graph.num_nodes):
            if node == graph.source or node == graph.sink:
                continue
            
            inflow = np.sum(flow_matrix[:, node])
            outflow = np.sum(flow_matrix[node, :])
            
            if abs(inflow - outflow) > 1e-6:
                return False
        
        return True
    
    def _get_best_known_value(self) -> float:
        """Ritorna il miglior valore conosciuto"""
        return max(self.convergence_history) if self.convergence_history else 0.0
