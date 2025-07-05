import random
import copy
import networkx as nx
import yaml
from typing import Dict, List, Tuple, Optional


class TabuMove:
    def __init__(self, modified_arcs: List[Tuple[int, int]], delta_values: List[float], move_type: str):
        self.modified_arcs = modified_arcs
        self.delta_values = delta_values
        self.move_type = move_type
    
    def get_reverse(self) -> 'TabuMove':
        reverse_deltas = [-delta for delta in self.delta_values]
        return TabuMove(self.modified_arcs.copy(), reverse_deltas, f"reverse_{self.move_type}")
    
    def __str__(self):
        return f"{self.move_type} on {len(self.modified_arcs)} arcs"


class TabuSearch:
    def __init__(self, graph: nx.DiGraph, source: int, sink: int, config_path: str = "config.yaml"):
        self.graph = graph
        self.source = source
        self.sink = sink
        
        self.config = self._load_config(config_path)
        self._initialize_parameters()
        
        self.edges = list(self.graph.edges())
        self.edge_capacities = {(u, v): self.graph[u][v]['capacity'] for u, v in self.edges}
        self.function_evaluations = 0
        
        # s = s0 (Initial solution)
        self.current_solution = self._get_initial_solution()
        self.best_solution = copy.deepcopy(self.current_solution)
        self.best_value = self._evaluate_solution(self.best_solution)
        
        # Initialize the tabu list, medium-term and long-term memories
        self._initialize_memory_structures()
        
        self.iteration = 0
        self.convergence_history = []
        self.no_improvement_count = 0
    
    def _load_config(self, config_path: str) -> dict:
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            # Configurazione di default se il file non esiste
            return {
                'tabu_search': {
                    'max_iterations': 20000,
                    'tabu_list_size': 50,
                    'aspiration_enabled': True,
                    'delta_step': 1.0,
                    'intensification_threshold': 50,
                    'diversification_threshold': 100,
                    'elite_size': 5
                }
            }
    
    def _initialize_parameters(self):
        tabu_config = self.config['tabu_search']
        self.max_iterations = tabu_config['max_iterations']
        self.tabu_list_size = tabu_config['tabu_list_size']
        self.aspiration_enabled = tabu_config.get('aspiration_enabled', True)
        self.delta_step = tabu_config.get('delta_step', 1.0)
        self.intensification_threshold = tabu_config.get('intensification_threshold', 50)
        self.diversification_threshold = tabu_config.get('diversification_threshold', 100)
        self.elite_size = tabu_config.get('elite_size', 5)
        
        # PARAMETRI EURISTICI RANDOMIZZATI per introdurre variabilità nei comportamenti
        # mantenendo comunque un controllo ragionevole
        randomization_factor = random.uniform(0.7, 1.3)
        self.delta_step *= randomization_factor
        
        # Randomizza soglie per comportamenti diversi tra run
        self.intensification_threshold = int(self.intensification_threshold * random.uniform(0.8, 1.2))
        self.diversification_threshold = int(self.diversification_threshold * random.uniform(0.8, 1.2))
        
        # Tabu list size randomizzata per esplorazioni diverse
        self.tabu_list_size = int(self.tabu_list_size * random.uniform(0.7, 1.3))
        self.tabu_list_size = max(10, min(200, self.tabu_list_size))
    
    def _initialize_memory_structures(self):
        self.tabu_list = []  # Short-term memory
        self.arc_frequency = {edge: 0 for edge in self.edges}  # Medium-term memory
        self.elite_solutions = []  # Long-term memory
    
    def _get_initial_solution(self) -> Dict[Tuple[int, int], float]:
        return {edge: 0.0 for edge in self.edges}
    
    def _evaluate_solution(self, solution: Dict[Tuple[int, int], float]) -> float:
        self.function_evaluations += 1
        return sum(solution.get((self.source, v), 0.0) for v in self.graph.successors(self.source))
    
    def _is_feasible(self, solution: Dict[Tuple[int, int], float]) -> bool:
        # Capacity constraints
        for (u, v), flow in solution.items():
            if flow < -1e-9 or flow > self.edge_capacities[(u, v)] + 1e-9:
                return False
        
        # Flow conservation for intermediate nodes
        for node in self.graph.nodes():
            if node == self.source or node == self.sink:
                continue
                
            flow_in = sum(solution.get((u, node), 0.0) for u in self.graph.predecessors(node))
            flow_out = sum(solution.get((node, v), 0.0) for v in self.graph.successors(node))
            
            if abs(flow_in - flow_out) > 1e-9:
                return False
        
        return True
    
    def _build_residual_graph(self, current_solution: Dict[Tuple[int, int], float]) -> nx.DiGraph:
        """Costruisce il grafo residuo"""
        residual_graph = nx.DiGraph()
        
        for (u, v), capacity in self.edge_capacities.items():
            current_flow = current_solution.get((u, v), 0.0)
            residual_capacity = capacity - current_flow
            
            if residual_capacity > 1e-9:
                residual_graph.add_edge(u, v, capacity=residual_capacity)
            
            if current_flow > 1e-9:
                residual_graph.add_edge(v, u, capacity=current_flow)
        
        return residual_graph
    
    def _random_walk_path(self, residual_graph: nx.DiGraph) -> Optional[List[Tuple[int, int]]]:
        """Random walk from source to sink"""
        current = self.source
        path = []
        visited = set()
        max_steps = min(100, len(residual_graph.nodes()) * 3)
        
        while current != self.sink and len(path) < max_steps:
            if current in visited and len(path) > 20:
                break
            visited.add(current)
            
            neighbors = list(residual_graph.successors(current))
            if not neighbors:
                break
            
            next_node = random.choice(neighbors)
            path.append((current, next_node))
            current = next_node
        
        return path if current == self.sink else None
    
    def _path_with_removed_edges(self, residual_graph: nx.DiGraph) -> Optional[List[Tuple[int, int]]]:
        """Find path after randomly removing some edges"""
        temp_graph = residual_graph.copy()
        edges = list(temp_graph.edges())
        
        if len(edges) > 20:
            num_remove = random.randint(5, min(len(edges) // 5, 30))
            edges_to_remove = random.sample(edges, num_remove)
            temp_graph.remove_edges_from(edges_to_remove)
        
        try:
            path = nx.shortest_path(temp_graph, self.source, self.sink)
            return [(path[i], path[i+1]) for i in range(len(path)-1)]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
    
    def _find_augmenting_path(self, current_solution: Dict[Tuple[int, int], float]) -> Optional[List[Tuple[int, int]]]:
        residual_graph = self._build_residual_graph(current_solution)
        
        try:
            # Force diversification with random strategies
            strategy = random.random()
            
            if strategy < 0.4:  # 40% - Random walk path
                return self._random_walk_path(residual_graph)
            elif strategy < 0.7:  # 30% - Remove random edges first
                return self._path_with_removed_edges(residual_graph)
            else:  # 30% - All shortest paths with random selection
                all_paths = list(nx.all_shortest_paths(residual_graph, self.source, self.sink))
                if not all_paths:
                    return None
                path = random.choice(all_paths)
                return [(path[i], path[i+1]) for i in range(len(path)-1)]
                
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
    
    def _find_random_shortest_path(self, current_solution: Dict[Tuple[int, int], float]) -> Optional[List[Tuple[int, int]]]:
        """Trova un percorso minimo scelto casualmente tra tutti i minimi"""
        residual_graph = self._build_residual_graph(current_solution)
        
        try:
            all_shortest = list(nx.all_shortest_paths(residual_graph, self.source, self.sink))
            if not all_shortest:
                return None
            
            # Selezione stocastica pesata sui percorsi
            weights = [1.0 / (len(path) + random.uniform(0, 2)) for path in all_shortest]
            selected_path = random.choices(all_shortest, weights=weights, k=1)[0]
            
            return [(selected_path[i], selected_path[i+1]) for i in range(len(selected_path)-1)]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
    
    def _find_weighted_random_walk(self, current_solution: Dict[Tuple[int, int], float]) -> Optional[List[Tuple[int, int]]]:
        """Random walk con bias verso il sink"""
        residual_graph = self._build_residual_graph(current_solution)
        current = self.source
        path = []
        visited_count = {}
        max_steps = min(100, len(self.graph.nodes()) * 2)
        
        while current != self.sink and len(path) < max_steps:
            neighbors = list(residual_graph.successors(current))
            if not neighbors:
                break
            
            # Penalizza nodi già visitati
            visited_penalty = visited_count.get(current, 0)
            if visited_penalty > 3:  # Evita cicli lunghi
                break
            
            # Calcola pesi euristici verso il sink
            weights = []
            for neighbor in neighbors:
                try:
                    # Distanza approssimata al sink (euristica)
                    distance_to_sink = nx.shortest_path_length(residual_graph, neighbor, self.sink)
                    weight = 1.0 / (distance_to_sink + 1) + random.uniform(0, 0.5)
                except:
                    weight = random.uniform(0.1, 1.0)
                weights.append(weight)
            
            # Selezione stocastica pesata
            next_node = random.choices(neighbors, weights=weights, k=1)[0]
            path.append((current, next_node))
            
            visited_count[current] = visited_count.get(current, 0) + 1
            current = next_node
        
        return path if current == self.sink else None
    
    def _find_path_with_perturbation(self, current_solution: Dict[Tuple[int, int], float]) -> Optional[List[Tuple[int, int]]]:
        """Trova percorso dopo perturbazione casuale del grafo residuo"""
        residual_graph = self._build_residual_graph(current_solution)
        
        # Perturbazione: rimuovi archi casuali
        edges = list(residual_graph.edges())
        if len(edges) > 10:
            num_remove = random.randint(1, min(len(edges) // 8, 15))
            edges_to_remove = random.sample(edges, num_remove)
            residual_graph.remove_edges_from(edges_to_remove)
        
        try:
            path = nx.shortest_path(residual_graph, self.source, self.sink)
            return [(path[i], path[i+1]) for i in range(len(path)-1)]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
    
    def _find_heuristic_biased_path(self, current_solution: Dict[Tuple[int, int], float]) -> Optional[List[Tuple[int, int]]]:
        """Percorso con bias verso archi con alta capacità residua"""
        residual_graph = self._build_residual_graph(current_solution)
        
        # Modifica pesi degli archi basandoti su euristiche
        for u, v, data in residual_graph.edges(data=True):
            residual_cap = data.get('capacity', 1.0)
            # Bias verso archi con alta capacità + randomizzazione
            heuristic_weight = 1.0 / (residual_cap + random.uniform(0.1, 1.0))
            residual_graph[u][v]['weight'] = heuristic_weight
        
        try:
            path = nx.shortest_path(residual_graph, self.source, self.sink, weight='weight')
            return [(path[i], path[i+1]) for i in range(len(path)-1)]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
    
    def _choose_heuristic_flow_increase(self, bottleneck: float) -> float:
        """Scelta euristica della quantità di flusso da aumentare - VERSIONE ENHANCED"""
        # Strategia stocastica per il flow increase con PIÙ VARIABILITÀ
        strategies = [
            bottleneck * 0.05,   # Passo piccolissimo
            bottleneck * 0.15,   # Passo molto piccolo
            bottleneck * 0.35,   # Passo piccolo  
            bottleneck * 0.55,   # Passo medio-piccolo
            bottleneck * 0.75,   # Passo medio-grande
            bottleneck * 0.95,   # Passo grande
            min(bottleneck, self.delta_step * 0.5),  # Passo limitato piccolo
            min(bottleneck, self.delta_step),        # Passo limitato standard
            min(bottleneck, self.delta_step * 1.5),  # Passo limitato grande
        ]
        
        # Probabilità più distribuite per aumentare variabilità
        weights = [0.15, 0.15, 0.15, 0.15, 0.15, 0.1, 0.05, 0.05, 0.05]
        chosen_increase = random.choices(strategies, weights=weights, k=1)[0]
        
        # Aggiunta di rumore casuale PIÙ AMPIO
        noise_factor = random.uniform(0.5, 1.8)  # Range più ampio
        result = chosen_increase * noise_factor
        
        # Occasionalmente usa valori molto piccoli per micro-exploration
        if random.random() < 0.1:  # 10% delle volte
            result *= random.uniform(0.1, 0.3)
            
        return result
    
    def _calculate_bottleneck(self, path: List[Tuple[int, int]], current_solution: Dict[Tuple[int, int], float]) -> float:
        """Calcola la capacità bottleneck di un percorso"""
        bottleneck = float('inf')
        for edge in path:
            if edge in self.edge_capacities:
                current_flow = current_solution.get(edge, 0.0)
                capacity = self.edge_capacities[edge]
                bottleneck = min(bottleneck, capacity - current_flow)
            else:
                reverse_edge = (edge[1], edge[0])
                if reverse_edge in current_solution:
                    current_flow = current_solution[reverse_edge]
                    bottleneck = min(bottleneck, current_flow)
        return bottleneck
    
    def _create_move_from_path(self, path: List[Tuple[int, int]], flow_increase: float) -> Tuple[List[Tuple[int, int]], List[float]]:
        """Crea una mossa da un percorso"""
        move_arcs = []
        move_deltas = []
        
        for edge in path:
            if edge in self.edge_capacities:
                move_arcs.append(edge)
                move_deltas.append(flow_increase)
            else:
                reverse_edge = (edge[1], edge[0])
                move_arcs.append(reverse_edge)
                move_deltas.append(-flow_increase)
        
        return move_arcs, move_deltas
    
    def _generate_stochastic_single_moves(self, current_solution: Dict[Tuple[int, int], float]) -> List[Tuple[TabuMove, Dict[Tuple[int, int], float]]]:
        """Genera mosse su singoli archi con selezione stocastica"""
        neighbors = []
        
        # Seleziona casualmente un sottoinsieme di archi da considerare
        candidate_edges = random.sample(self.edges, min(len(self.edges), 50))
        
        for edge in candidate_edges:
            current_flow = current_solution.get(edge, 0.0)
            capacity = self.edge_capacities[edge]
            
            # Probabilità di considerare questo arco
            if random.random() < 0.4:  # 40% probabilità
                continue
            
            # Mosse di incremento con step stocastico
            if current_flow < capacity - 1e-9:
                step_factor = random.choice([0.1, 0.2, 0.5, 1.0])
                step = min(capacity - current_flow, self.delta_step * step_factor)
                
                if step > 1e-6:
                    move = TabuMove([edge], [step], f"stochastic_increase_{step_factor}")
                    new_solution = self._apply_move(current_solution, move)
                    if new_solution and self._is_feasible(new_solution):
                        neighbors.append((move, new_solution))
            
            # Mosse di decremento con step stocastico
            if current_flow > 1e-9:
                step_factor = random.choice([0.1, 0.3, 0.7, 1.0])
                step = min(current_flow, self.delta_step * step_factor)
                
                move = TabuMove([edge], [-step], f"stochastic_decrease_{step_factor}")
                new_solution = self._apply_move(current_solution, move)
                if new_solution and self._is_feasible(new_solution):
                    neighbors.append((move, new_solution))
        
        return neighbors
    
    def _generate_combinatorial_moves(self, current_solution: Dict[Tuple[int, int], float]) -> List[Tuple[TabuMove, Dict[Tuple[int, int], float]]]:
        """Genera mosse che modificano multiple archi simultaneamente"""
        neighbors = []
        
        for attempt in range(5):  # Limitato per non rallentare troppo
            # Seleziona 2-4 archi casuali
            num_arcs = random.randint(2, 4)
            selected_edges = random.sample(self.edges, min(len(self.edges), num_arcs))
            
            move_arcs = []
            move_deltas = []
            
            for edge in selected_edges:
                current_flow = current_solution.get(edge, 0.0)
                capacity = self.edge_capacities[edge]
                
                # Scelta casuale: incremento o decremento
                if random.random() < 0.5 and current_flow < capacity - 1e-9:
                    # Incremento
                    delta = random.uniform(0.1, min(capacity - current_flow, self.delta_step))
                    move_arcs.append(edge)
                    move_deltas.append(delta)
                elif current_flow > 1e-9:
                    # Decremento
                    delta = random.uniform(0.1, min(current_flow, self.delta_step))
                    move_arcs.append(edge)
                    move_deltas.append(-delta)
            
            if move_arcs:
                move = TabuMove(move_arcs, move_deltas, f"combinatorial_{attempt}")
                new_solution = self._apply_move(current_solution, move)
                if new_solution and self._is_feasible(new_solution):
                    neighbors.append((move, new_solution))
        
        return neighbors
    
    def _select_stochastic_neighborhood(self, neighbors: List[Tuple[TabuMove, Dict[Tuple[int, int], float]]]) -> List[Tuple[TabuMove, Dict[Tuple[int, int], float]]]:
        """Selezione finale stocastica del vicinato"""
        if not neighbors:
            return []
        
        # Shuffle per ordine casuale
        random.shuffle(neighbors)
        
        # Seleziona un sottoinsieme casuale del vicinato
        max_neighbors = min(len(neighbors), random.randint(20, 60))
        selected = neighbors[:max_neighbors]
        
        return selected
    
    def _generate_neighborhood(self, current_solution: Dict[Tuple[int, int], float]) -> List[Tuple[TabuMove, Dict[Tuple[int, int], float]]]:
        """
        EURISTIC NEIGHBORHOOD GENERATION:
        Genera un vicinato stocastico invece di deterministico per introdurre variabilità tra i run
        """
        neighbors = []
        
        # Strategia 1: Path-based moves con selezione stocastica
        for attempt in range(20):
            # Scelta stocastica della strategia di ricerca percorso
            strategy_prob = random.random()
            
            if strategy_prob < 0.3:  # 30% - Percorso minimo casuale
                path = self._find_random_shortest_path(current_solution)
            elif strategy_prob < 0.5:  # 20% - Random walk ponderato
                path = self._find_weighted_random_walk(current_solution)
            elif strategy_prob < 0.7:  # 20% - Percorso con rimozione archi
                path = self._find_path_with_perturbation(current_solution)
            else:  # 30% - Percorso con bias euristico
                path = self._find_heuristic_biased_path(current_solution)
            
            if not path:
                continue
            
            # Step size stocastico invece di fisso
            bottleneck = self._calculate_bottleneck(path, current_solution)
            
            if bottleneck > 1e-9:
                # Scelta euristica del flow increase
                flow_increase = self._choose_heuristic_flow_increase(bottleneck)
                
                if flow_increase < 1e-6:
                    continue
                
                move_arcs, move_deltas = self._create_move_from_path(path, flow_increase)
                move = TabuMove(move_arcs, move_deltas, f"heuristic_path_{attempt}")
                new_solution = self._apply_move(current_solution, move)
                
                if new_solution and self._is_feasible(new_solution):
                    neighbors.append((move, new_solution))
        
        # Strategia 2: Single-arc moves stocastici
        neighbors.extend(self._generate_stochastic_single_moves(current_solution))
        
        # Strategia 3: Multi-arc moves combinatori
        neighbors.extend(self._generate_combinatorial_moves(current_solution))
        
        # Selezione finale stocastica del vicinato
        return self._select_stochastic_neighborhood(neighbors)
    
    def _apply_move(self, solution: Dict[Tuple[int, int], float], move: TabuMove) -> Optional[Dict[Tuple[int, int], float]]:
        new_solution = copy.deepcopy(solution)
        
        for edge, delta in zip(move.modified_arcs, move.delta_values):
            if edge in new_solution:
                new_solution[edge] += delta
            else:
                new_solution[edge] = delta
                
            new_solution[edge] = max(0.0, new_solution[edge])
        
        return new_solution
    
    def _is_move_tabu(self, move: TabuMove) -> bool:
        reverse_move = move.get_reverse()
        
        for tabu_move in self.tabu_list:
            if (set(tabu_move.modified_arcs) == set(reverse_move.modified_arcs) and
                len(tabu_move.delta_values) == len(reverse_move.delta_values)):
                
                all_reverse = True
                for t_delta, r_delta in zip(tabu_move.delta_values, reverse_move.delta_values):
                    if abs(t_delta - r_delta) > 1e-9:
                        all_reverse = False
                        break
                
                if all_reverse:
                    return True
        
        return False
    
    def _aspiration_criterion(self, move: TabuMove, solution_value: float) -> bool:
        return self.aspiration_enabled and solution_value > self.best_value
    
    def _heuristic_selection_from_candidates(self, candidates: List[Dict]) -> Tuple[TabuMove, Dict[Tuple[int, int], float], float]:
        """Selezione euristica tra candidati - VERSIONE ENHANCED per più variabilità"""
        # Ordina per valore
        candidates.sort(key=lambda x: x['value'], reverse=True)
        
        # Strategia euristica di selezione con PIÙ VARIABILITÀ
        selection_strategy = random.random()
        
        if selection_strategy < 0.25:  # 25% - Greedy: prendi il migliore
            best = candidates[0]
            return best['move'], best['solution'], best['value']
        
        elif selection_strategy < 0.45:  # 20% - Top-k selection stocastica
            k = min(len(candidates), random.randint(3, 12))  # Range più ampio
            top_k = candidates[:k]
            # Probabilità decrescenti per i top-k
            weights = [1.0 / (i + 1) for i in range(len(top_k))]
            selected = random.choices(top_k, weights=weights, k=1)[0]
            return selected['move'], selected['solution'], selected['value']
        
        elif selection_strategy < 0.65:  # 20% - Selezione completamente casuale
            selected = random.choice(candidates)
            return selected['move'], selected['solution'], selected['value']
        
        elif selection_strategy < 0.80:  # 15% - Selezione biased verso soluzioni intermedie
            mid_range_start = max(0, len(candidates) // 4)
            mid_range_end = min(len(candidates), 3 * len(candidates) // 4)
            if mid_range_end > mid_range_start:
                mid_candidates = candidates[mid_range_start:mid_range_end]
                selected = random.choice(mid_candidates)
                return selected['move'], selected['solution'], selected['value']
            else:
                selected = random.choice(candidates)
                return selected['move'], selected['solution'], selected['value']
        
        else:  # 20% - Selezione con bias verso diversità
            # Prendi uno dei candidati dalla seconda metà (meno buoni)
            second_half_start = len(candidates) // 2
            if second_half_start < len(candidates):
                worse_candidates = candidates[second_half_start:]
                selected = random.choice(worse_candidates)
                return selected['move'], selected['solution'], selected['value']
            else:
                selected = random.choice(candidates)
                return selected['move'], selected['solution'], selected['value']
    
    def _select_best_neighbor_heuristic(self, neighbors: List[Tuple[TabuMove, Dict[Tuple[int, int], float]]]) -> Tuple[Optional[TabuMove], Optional[Dict[Tuple[int, int], float]], float]:
        """
        EURISTIC NEIGHBOR SELECTION:
        Selezione euristica del miglior vicino invece di selezione deterministica
        """
        if not neighbors:
            return None, None, float('-inf')
        
        # Valuta tutti i vicini
        neighbor_evaluations = []
        for move, neighbor in neighbors:
            neighbor_value = self._evaluate_solution(neighbor)
            is_tabu = self._is_move_tabu(move)
            satisfies_aspiration = self._aspiration_criterion(move, neighbor_value)
            
            neighbor_evaluations.append({
                'move': move,
                'solution': neighbor,
                'value': neighbor_value,
                'is_tabu': is_tabu,
                'satisfies_aspiration': satisfies_aspiration,
                'is_admissible': not is_tabu or satisfies_aspiration
            })
        
        # Strategia 1: Cerca vicini ammissibili
        admissible_neighbors = [n for n in neighbor_evaluations if n['is_admissible']]
        
        if admissible_neighbors:
            return self._heuristic_selection_from_candidates(admissible_neighbors)
        
        # Strategia 2: Se nessun vicino ammissibile, selezione euristica tra tutti
        return self._heuristic_selection_from_candidates(neighbor_evaluations)
    
    def _update_memory_structures(self, move: TabuMove, solution: Dict[Tuple[int, int], float], solution_value: float):
        # Update tabu list, aspiration conditions, medium and long term memories
        self.tabu_list.append(move)
        if len(self.tabu_list) > self.tabu_list_size:
            self.tabu_list.pop(0)
        
        # Medium-term memory: arc frequency
        for edge in move.modified_arcs:
            if edge in self.arc_frequency:
                self.arc_frequency[edge] += 1
        
        # Long-term memory: elite solutions
        self._update_elite_solutions(solution, solution_value)
    
    def _update_elite_solutions(self, solution: Dict[Tuple[int, int], float], value: float):
        self.elite_solutions.append((copy.deepcopy(solution), value))
        self.elite_solutions.sort(key=lambda x: x[1], reverse=True)
        if len(self.elite_solutions) > self.elite_size:
            self.elite_solutions = self.elite_solutions[:self.elite_size]
    
    def _intensification_criterion(self) -> bool:
        return self.no_improvement_count >= self.intensification_threshold
    
    def _diversification_criterion(self) -> bool:
        return self.no_improvement_count >= self.diversification_threshold
    
    def _apply_intensification(self):
        # Intensification strategy
        self.delta_step *= 0.8
        self.no_improvement_count = max(0, self.no_improvement_count // 2)
    
    def _apply_diversification(self):
        """
        AGGRESSIVE DIVERSIFICATION ENHANCED:
        Diversificazione aggressiva potenziata per creare maggiore variabilità
        """
        diversification_strategy = random.random()
        
        if diversification_strategy < 0.3:  # 30% - Restart da soluzione elite
            if self.elite_solutions:
                elite_solution, _ = random.choice(self.elite_solutions)
                self.current_solution = copy.deepcopy(elite_solution)
                
                # Perturbazioni casuali MOLTO aggressive
                num_perturbations = random.randint(20, min(len(self.edges) // 2, 100))
                for edge in random.sample(self.edges, num_perturbations):
                    if self.current_solution[edge] > 1e-9:
                        reduction_factor = random.uniform(0.1, 0.9)  # Più aggressivo
                        self.current_solution[edge] *= (1 - reduction_factor)
        
        elif diversification_strategy < 0.5:  # 20% - Restart parziale molto aggressivo
            # Mantieni solo una frazione piccola del flusso corrente
            keep_fraction = random.uniform(0.1, 0.4)  # Più aggressivo
            for edge in self.edges:
                if self.current_solution[edge] > 1e-9:
                    if random.random() < (1 - keep_fraction):
                        reduction = random.uniform(0.7, 1.0)  # Riduzioni maggiori
                        self.current_solution[edge] *= (1 - reduction)
        
        elif diversification_strategy < 0.7:  # 20% - Restart completo con inizializzazione casuale
            self.current_solution = self._get_initial_solution()
            
            # Applica un flusso casuale iniziale più sostanzioso
            for _ in range(random.randint(10, 30)):
                random_edge = random.choice(self.edges)
                if random_edge[0] == self.source:  # Solo archi dalla sorgente
                    random_flow = random.uniform(0.5, min(3.0, self.edge_capacities[random_edge] * 0.3))
                    self.current_solution[random_edge] = random_flow
        
        else:  # 30% - Perturbazione dell'ottimo corrente
            # NUOVA STRATEGIA: perturba la soluzione corrente mantenendo fattibilità
            if self.best_value > 0:
                # Prendi la miglior soluzione e perturbala leggermente
                self.current_solution = copy.deepcopy(self.best_solution)
                
                # Applica micro-perturbazioni per esplorare intorno all'ottimo
                for _ in range(random.randint(5, 15)):
                    edge = random.choice(self.edges)
                    if edge in self.current_solution and self.current_solution[edge] > 1e-6:
                        # Perturbazione molto piccola
                        perturbation = random.uniform(-0.1, 0.1) * self.current_solution[edge]
                        new_value = self.current_solution[edge] + perturbation
                        new_value = max(0.0, min(new_value, self.edge_capacities[edge]))
                        self.current_solution[edge] = new_value
        
        # Reset memorie e parametri con randomizzazione
        self.tabu_list.clear()
        self.no_improvement_count = 0
        
        # Randomizza MOLTO di più i parametri per il prossimo ciclo
        base_delta = self.config['tabu_search'].get('delta_step', 1.0)
        self.delta_step = base_delta * random.uniform(0.3, 2.0)  # Range molto più ampio
        
        # Randomizza anche la tabu list size
        base_tabu_size = self.config['tabu_search'].get('tabu_list_size', 100)
        self.tabu_list_size = int(base_tabu_size * random.uniform(0.5, 1.8))
        self.tabu_list_size = max(5, min(300, self.tabu_list_size))
    
    def _stopping_criteria_satisfied(self) -> bool:
        """
        STOPPING CRITERIA come specificato dal progetto + arresto intelligente:
        1. Raggiunto max_iterations (2×10^4)
        2. Raggiunto il massimo flusso (verifica di ottimalità)
        """
        # Criterio 1: Max iterazioni raggiunto
        if self.iteration >= self.max_iterations:
            return True
        
        # Criterio 2: Ottimo teorico raggiunto
        theoretical_max = min(
            sum(self.edge_capacities.get((self.source, v), 0) for v in self.graph.successors(self.source)),
            sum(self.edge_capacities.get((u, self.sink), 0) for u in self.graph.predecessors(self.sink))
        )
        
        # Verifica di ottimalità: quando il massimo flusso trovato coincide con 
        # la somma delle capacità degli archi che escono dalla sorgente o che entrano nel pozzo
        if abs(self.best_value - theoretical_max) < 1e-3:
            return True
        
        return False 
    
    def solve(self) -> Tuple[Dict[Tuple[int, int], float], float, List[float]]:
        """
        MAIN TABU SEARCH ALGORITHM:
        Implementazione euristica del TabuSearch per il problema del massimo flusso
        con criteri di arresto conformi alle specifiche del progetto
        """
        # Main Tabu Search algorithm following Algorithm 2.9
        while not self._stopping_criteria_satisfied():
            self.iteration += 1
            
            # Generate neighborhood N(s) - EURISTIC VERSION
            neighbors = self._generate_neighborhood(self.current_solution)
            
            # Select best neighbor s' ∈ N(s) - EURISTIC SELECTION
            best_move, best_neighbor, best_neighbor_value = self._select_best_neighbor_heuristic(neighbors)
            
            # s = s'
            if best_move is not None:
                self.current_solution = best_neighbor
                current_value = best_neighbor_value
                
                # Update best solution if improved
                if current_value > self.best_value:
                    self.best_solution = copy.deepcopy(self.current_solution)
                    self.best_value = current_value
                    self.no_improvement_count = 0
                else:
                    self.no_improvement_count += 1
                
                # Update tabu list, aspiration conditions, medium and long term memories
                self._update_memory_structures(best_move, self.current_solution, current_value)
                self.convergence_history.append(self.best_value)
                
                # If intensification_criterion holds Then intensification
                if self._intensification_criterion():
                    self._apply_intensification()
                
                # If diversification_criterion holds Then diversification
                if self._diversification_criterion():
                    self._apply_diversification()
            
            else:
                # No valid moves found - force diversification
                self._apply_diversification()
                self.convergence_history.append(self.best_value)
        
        # Output: Best solution found
        return self.best_solution, self.best_value, self.convergence_history
    
    def get_function_evaluations(self) -> int:
        return self.function_evaluations
    
    def get_memory_statistics(self) -> dict:
        return {
            'tabu_list_size': len(self.tabu_list),
            'most_frequent_arcs': sorted(self.arc_frequency.items(), key=lambda x: x[1], reverse=True)[:5],
            'elite_solutions_count': len(self.elite_solutions),
            'elite_values': [value for _, value in self.elite_solutions]
        }


# ESEMPIO DI UTILIZZO E TEST
if __name__ == "__main__":
    # Creazione di un grafo di test
    def create_test_graph():
        G = nx.DiGraph()
        # Aggiungi nodi
        nodes = [0, 1, 2, 3, 4, 5]  # 0=source, 5=sink
        G.add_nodes_from(nodes)
        
        # Aggiungi archi con capacità
        edges = [
            (0, 1, 10), (0, 2, 8),
            (1, 2, 2), (1, 3, 5), (1, 4, 8),
            (2, 4, 10),
            (3, 5, 10),
            (4, 3, 3), (4, 5, 10)
        ]
        
        for u, v, capacity in edges:
            G.add_edge(u, v, capacity=capacity)
        
        return G
    
    # Test con grafo di esempio
    test_graph = create_test_graph()
    
    # Configurazione di default per test
    config = {
        'tabu_search': {
            'max_iterations': 1000,
            'tabu_list_size': 20,
            'aspiration_enabled': True,
            'delta_step': 1.0,
            'intensification_threshold': 50,
            'diversification_threshold': 100,
            'elite_size': 5
        }
    }
    
    # Salva configurazione temporanea
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name
    
    try:
        # Inizializza e risolvi
        tabu_search = TabuSearch(test_graph, source=0, sink=5, config_path=config_path)
        
        print("=== TABU SEARCH EURISTICO PER MASSIMO FLUSSO ===")
        print(f"Grafo: {len(test_graph.nodes())} nodi, {len(test_graph.edges())} archi")
        print(f"Source: {tabu_search.source}, Sink: {tabu_search.sink}")
        print(f"Parametri: max_iter={tabu_search.max_iterations}, tabu_size={tabu_search.tabu_list_size}")
        print("\nAvvio risoluzione...")
        
        # Risolvi il problema
        best_solution, best_value, convergence = tabu_search.solve()
        
        print(f"\n=== RISULTATI ===")
        print(f"Miglior flusso trovato: {best_value:.4f}")
        print(f"Iterazioni eseguite: {tabu_search.iteration}")
        print(f"Valutazioni funzione obiettivo: {tabu_search.get_function_evaluations()}")
        print(f"Convergenza: {len(convergence)} valori registrati")
        
        # Mostra statistiche memoria
        stats = tabu_search.get_memory_statistics()
        print(f"\n=== STATISTICHE MEMORIA ===")
        print(f"Dimensione lista tabu: {stats['tabu_list_size']}")
        print(f"Soluzioni elite: {stats['elite_solutions_count']}")
        if stats['elite_values']:
            print(f"Valori elite: {[f'{v:.2f}' for v in stats['elite_values'][:3]]}")
        
        # Mostra alcuni archi con flusso positivo
        print(f"\n=== SOLUZIONE FINALE ===")
        active_flows = [(edge, flow) for edge, flow in best_solution.items() if flow > 1e-6]
        active_flows.sort(key=lambda x: x[1], reverse=True)
        
        print("Archi attivi (con flusso > 0):")
        for edge, flow in active_flows[:10]:  # Mostra i primi 10
            capacity = tabu_search.edge_capacities[edge]
            print(f"  {edge}: flusso={flow:.4f}, capacità={capacity}")
        
        # Verifica feasibility
        if tabu_search._is_feasible(best_solution):
            print("\n✓ Soluzione FEASIBILE")
        else:
            print("\n✗ Soluzione NON FEASIBILE")
            
    finally:
        # Pulisci file temporaneo
        os.unlink(config_path)