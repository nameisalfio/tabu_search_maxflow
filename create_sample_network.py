#!/usr/bin/env python3
"""
Script per creare un network di esempio per testing
"""

def create_sample_network():
    """Crea un network di esempio semplice"""
    
    # Network semplice: 6 nodi, source=0, sink=5
    content = """6 0 5
0 1 10
0 2 8
1 3 5
1 4 8
2 3 3
2 4 2
3 -1 10
4 -1 10"""
    
    with open("data/networks/network_sample.txt", 'w') as f:
        f.write(content)
    
    print("✅ Sample network created: data/networks/network_sample.txt")
    print("   6 nodes, source=0, sink=5")
    print("   Expected max flow: ~15")

if __name__ == "__main__":
    create_sample_network()
