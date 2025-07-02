"""
Network Reader Module
Legge e parsa i file network per il Maximum Flow Problem
"""

import numpy as np
from typing import Optional, List, Tuple

class NetworkGraph:
    """Rappresenta un grafo di rete per il maximum flow"""
    
    def __init__(self, num_nodes: int, source: int, sink: int, edges: List[Tuple[int, int, int]]):
        self.num_nodes = num_nodes
        self.source = source
        self.sink = sink
        self.edges = edges
        
        # Costruisci matrici
        self.capacity_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
        self.adjacency_list = {i: [] for i in range(num_nodes)}
        
        for u, v, capacity in edges:
            if v == -1:  # Sink marker
                v = sink
            self.capacity_matrix[u][v] = capacity
            self.adjacency_list[u].append(v)
    
    def __str__(self):
        return f"NetworkGraph(nodes={self.num_nodes}, source={self.source}, sink={self.sink}, edges={len(self.edges)})"

class NetworkReader:
    """Legge file network e crea oggetti NetworkGraph"""
    
    def read_network_file(self, filepath: str) -> Optional[NetworkGraph]:
        """
        Legge un file network
        
        Formato file:
        Prima riga: num_nodes source sink
        Righe successive: from_node to_node capacity
        """
        try:
            with open(filepath, 'r') as file:
                lines = [line.strip() for line in file.readlines() if line.strip()]
            
            # Parse prima riga
            first_line = lines[0].split()
            num_nodes = int(first_line[0])
            source = int(first_line[1])
            sink = int(first_line[2])
            
            # Parse archi
            edges = []
            for line in lines[1:]:
                parts = line.split()
                if len(parts) == 3:
                    u, v, capacity = int(parts[0]), int(parts[1]), int(parts[2])
                    edges.append((u, v, capacity))
            
            print(f"✅ Network loaded: {num_nodes} nodes, {len(edges)} edges")
            return NetworkGraph(num_nodes, source, sink, edges)
            
        except Exception as e:
            print(f"❌ Error reading {filepath}: {e}")
            return None

class FlowSolution:
    """Rappresenta una soluzione di flusso"""
    
    def __init__(self, flow_matrix: np.ndarray, flow_value: float):
        self.flow_matrix = flow_matrix
        self.flow_value = flow_value
        self.is_valid = True
        
    def copy(self):
        """Crea una copia della soluzione"""
        return FlowSolution(self.flow_matrix.copy(), self.flow_value)
