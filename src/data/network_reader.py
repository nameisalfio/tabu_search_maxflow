import networkx as nx
import os
from typing import Tuple, Dict, Any


class NetworkData:
    
    def __init__(self, graph: nx.DiGraph, source: int, sink: int, info: Dict[str, Any]):
        self.graph = graph
        self.source = source
        self.sink = sink
        self.info = info
        
    def is_optimal_flow(self, flow_value: float, tolerance: float = 1e-6) -> bool:
        """
        Check if a flow value is optimal by comparing with theoretical bounds.
        A flow is optimal when it equals the sum of capacities from source OR to sink.
        """
        source_capacity = self.info['total_source_capacity']
        sink_capacity = self.info['total_sink_capacity']
        
        return (abs(flow_value - source_capacity) <= tolerance or 
                abs(flow_value - sink_capacity) <= tolerance)
    
    def get_max_possible_flow(self) -> float:
        """Return the theoretical maximum possible flow."""
        return min(self.info['total_source_capacity'], self.info['total_sink_capacity'])


def read_network(filepath: str, verbose: bool = True) -> NetworkData:
    """
    Read a network file and return network data.
    
    File format:
    - Line 1: number of nodes
    - Line 2: number of edges  
    - Line 3: source node (negative, e.g., -2)
    - Line 4: sink node (negative, e.g., -1)
    - Following lines: triples (u, v, capacity)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} not found")
    
    filename = os.path.basename(filepath)
    
    if verbose:
        print(f"Loading {filename}")
    
    # Read file
    with open(filepath, 'r') as file:
        lines = [line.strip() for line in file.readlines() if line.strip()]
    
    if len(lines) < 4:
        raise ValueError(f"Malformed file: need at least 4 lines, found {len(lines)}")

    try:
        num_nodes = int(lines[0])
        num_edges = int(lines[1])
        source = int(lines[2])  
        sink = int(lines[3])    
    except ValueError as e:
        raise ValueError(f"Error parsing header: {e}")
    
    # Create directed graph (empty initially)
    graph = nx.DiGraph()
    
    nodes_seen = set()
    edges_loaded = 0
    total_source_capacity = 0.0
    total_sink_capacity = 0.0
    
    # Parse edges and add nodes/edges dynamically
    for line in lines[4:]:
        if not line:
            continue
            
        try:
            parts = line.split()
            if len(parts) != 3:
                continue
                
            u, v, capacity = int(parts[0]), int(parts[1]), float(parts[2])
            
            # Basic validation
            if capacity <= 0:
                continue
            
            # Add nodes 
            if u not in nodes_seen:
                graph.add_node(u)
                nodes_seen.add(u)
            
            if v not in nodes_seen:
                graph.add_node(v)
                nodes_seen.add(v)
            
            # Add edge
            graph.add_edge(u, v, capacity=capacity)
            edges_loaded += 1
            
            # Calculate total capacities for optimality bounds
            if u == source:
                total_source_capacity += capacity
            if v == sink:
                total_sink_capacity += capacity
                
        except (ValueError, IndexError) as e:
            continue
    
    # Ensure source and sink are in the graph
    if source not in nodes_seen:
        graph.add_node(source)
        nodes_seen.add(source)
    
    if sink not in nodes_seen:
        graph.add_node(sink)
        nodes_seen.add(sink)
    
    # Check essential connectivity
    if graph.out_degree(source) == 0:
        raise ValueError(f"Source {source} has no outgoing edges!")
    if graph.in_degree(sink) == 0:
        raise ValueError(f"Sink {sink} has no incoming edges!")
    
    # Create info dictionary 
    info = {
        'filename': filename,
        'num_nodes_declared': num_nodes,
        'num_edges_declared': num_edges,
        'num_nodes_actual': graph.number_of_nodes(),
        'num_edges_actual': edges_loaded,
        'source': source,
        'sink': sink,
        'total_source_capacity': total_source_capacity,
        'total_sink_capacity': total_sink_capacity,
        'max_possible_flow': min(total_source_capacity, total_sink_capacity),
        'nodes_seen': len(nodes_seen)
    }
    
    if verbose:
        print_network_statistics(info)
    
    return NetworkData(graph, source, sink, info)


def print_network_statistics(info: Dict[str, Any]):
    """Print minimal network statistics focused on flow information."""
    
    print(f"  Nodes: {info['num_nodes_actual']}, Edges: {info['num_edges_actual']}")
    print(f"  Source {info['source']}: capacity out = {info['total_source_capacity']:.2f}")
    print(f"  Sink {info['sink']}: capacity in = {info['total_sink_capacity']:.2f}")
    print(f"  Max possible flow: {info['max_possible_flow']:.2f}")
    

def create_flow_dict(graph: nx.DiGraph, initial_value: float = 0.0) -> Dict[Tuple[int, int], float]:
    """Create a flow dictionary for all edges in the graph."""
    return {(u, v): initial_value for u, v, _ in graph.edges(data=True)}


def validate_flow(network_data: NetworkData, flow_dict: Dict[Tuple[int, int], float]) -> Tuple[bool, float, list]:
    """
    Validate a flow solution for a network.
    """
    graph = network_data.graph
    source = network_data.source
    sink = network_data.sink
    errors = []
    
    # Capacity and non-negativity constraints
    for (u, v), flow in flow_dict.items():
        if (u, v) not in graph.edges():
            errors.append(f"Edge ({u}, {v}) does not exist")
            continue
            
        if flow < 0:
            errors.append(f"Negative flow on ({u}, {v}): {flow:.6f}")
            
        capacity = graph[u][v]['capacity']
        if flow > capacity + 1e-9:
            errors.append(f"Capacity violated on ({u}, {v}): {flow:.6f} > {capacity:.6f}")
    
    # Flow conservation
    for node in graph.nodes():
        if node == source or node == sink:
            continue
            
        flow_in = sum(flow_dict.get((u, node), 0.0) for u in graph.predecessors(node))
        flow_out = sum(flow_dict.get((node, v), 0.0) for v in graph.successors(node))
        
        if abs(flow_in - flow_out) > 1e-9:
            errors.append(f"Flow conservation violated at node {node}: in={flow_in:.6f} != out={flow_out:.6f}")
    
    # Calculate flow value
    flow_value = sum(flow_dict.get((source, v), 0.0) for v in graph.successors(source))
    
    return len(errors) == 0, flow_value, errors


def compute_reference_max_flow(network_data: NetworkData, verbose: bool = False) -> float:
    """
    Compute reference max flow using NetworkX (use only for final comparison).
    This function should ONLY be called after experiments are complete.
    """
    try:
        max_flow_value, _ = nx.maximum_flow(
            network_data.graph, 
            network_data.source, 
            network_data.sink, 
            capacity='capacity'
        )
        if verbose:
            print(f"Reference max flow (NetworkX): {max_flow_value:.2f}")
        return max_flow_value
    except Exception as e:
        if verbose:
            print(f"Error computing reference max flow: {e}")
        return 0.0

