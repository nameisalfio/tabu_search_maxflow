# src/data/network_reader.py
import os
import networkx as nx
from typing import Dict, Any

class NetworkData:
    """A container for the graph, source, sink, and metadata."""
    def __init__(self, graph: nx.DiGraph, source: int, sink: int, info: Dict[str, Any]):
        self.graph = graph
        self.source = source
        self.sink = sink
        self.info = info

    def get_max_possible_flow(self) -> float:
        """Returns the theoretical maximum flow, capped by source/sink capacity."""
        return min(self.info.get('total_source_capacity', 0), self.info.get('total_sink_capacity', 0))

def read_network(filepath: str) -> NetworkData:
    """
    Reads a network file and returns a NetworkData object.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, 'r') as file:
        lines = [line.strip() for line in file.readlines() if line.strip()]

    source = int(lines[2])
    sink = int(lines[3])

    graph = nx.DiGraph()
    total_source_capacity = 0.0
    total_sink_capacity = 0.0

    for line in lines[4:]:
        u, v, capacity = line.split()
        u, v, capacity = int(u), int(v), float(capacity)
        if capacity > 0:
            graph.add_edge(u, v, capacity=capacity)
            if u == source:
                total_source_capacity += capacity
            if v == sink:
                total_sink_capacity += capacity
    
    if source not in graph: graph.add_node(source)
    if sink not in graph: graph.add_node(sink)

    info = {
        'filename': os.path.basename(filepath),
        'num_nodes': graph.number_of_nodes(),
        'num_edges': graph.number_of_edges(),
        'source': source,
        'sink': sink,
        'total_source_capacity': total_source_capacity,
        'total_sink_capacity': total_sink_capacity,
    }
    return NetworkData(graph, source, sink, info)

def compute_reference_max_flow(network_data: NetworkData) -> float:
    """Computes the exact maximum flow using a standard algorithm for validation."""
    if not network_data.graph.has_node(network_data.source) or not network_data.graph.has_node(network_data.sink):
        return 0.0
    try:
        max_flow_value, _ = nx.maximum_flow(
            network_data.graph, 
            network_data.source, 
            network_data.sink, 
            capacity='capacity'
        )
        return max_flow_value
    except nx.NetworkXError:
        return 0.0