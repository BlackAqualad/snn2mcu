"""
Utility script to open and inspect NIR files.
This script loads a .nir file and displays its structure, nodes, and edges.
"""

import nir
import numpy as np

# Load the NIR file
nir_file = 'snntorch_snn_232.nir'
print(f"Loading NIR file: {nir_file}\n")

try:
    # Read the NIR graph
    nir_graph = nir.read(nir_file)
    
    print("=" * 70)
    print("NIR GRAPH STRUCTURE")
    print("=" * 70)
    
    # Display nodes
    print("\n--- NODES ---")
    for node_name, node in nir_graph.nodes.items():
        print(f"\n{node_name}:")
        print(f"  Type: {type(node).__name__}")
        
        # Display node-specific information
        if isinstance(node, nir.Input):
            print(f"  Input type: {node.input_type}")
        
        elif isinstance(node, nir.Output):
            print(f"  Output type: {node.output_type}")
        
        elif isinstance(node, nir.Affine):
            print(f"  Weight shape: {node.weight.shape}")
            print(f"  Bias shape: {node.bias.shape}")
            print(f"  Weight (first 3x3 if available):\n{node.weight[:3, :3]}")
            if np.any(node.bias != 0):
                print(f"  Bias (first 3): {node.bias[:3]}")
        
        elif isinstance(node, nir.LIF):
            print(f"  Number of neurons: {len(node.tau)}")
            print(f"  Tau: {node.tau[0]:.4f} (all neurons)" if np.all(node.tau == node.tau[0]) else f"  Tau (first 3): {node.tau[:3]}")
            print(f"  Threshold: {node.v_threshold[0]:.4f}")
            print(f"  V_leak: {node.v_leak[0]:.4f}")
            print(f"  V_reset: {node.v_reset[0]:.4f}")
            print(f"  Resistance: {node.r[0]:.4f}")
    
    # Display edges
    print("\n--- EDGES (Network Connections) ---")
    for i, edge in enumerate(nir_graph.edges, 1):
        print(f"{i}. {edge[0]} -> {edge[1]}")
    
    print("\n" + "=" * 70)
    print("NETWORK SUMMARY")
    print("=" * 70)
    print(f"Total nodes: {len(nir_graph.nodes)}")
    print(f"Total edges: {len(nir_graph.edges)}")
    
    # Count different node types
    node_types = {}
    for node in nir_graph.nodes.values():
        node_type = type(node).__name__
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    print("\nNode types:")
    for node_type, count in node_types.items():
        print(f"  {node_type}: {count}")
    
    print("\n" + "=" * 70)
    print("GRAPH VISUALIZATION")
    print("=" * 70)
    
    # Create a text-based graph visualization
    print("\nNetwork Flow Diagram:")
    print()
    
    # Build adjacency list for visualization
    adjacency = {}
    for src, dst in nir_graph.edges:
        if src not in adjacency:
            adjacency[src] = []
        adjacency[src].append(dst)
    
    # Perform a simple traversal starting from input
    def print_graph_node(node_name, visited=None, indent=0, is_last_child=True, prefix=""):
        if visited is None:
            visited = set()
        
        if node_name in visited:
            # Show reference to already visited node (recurrent connection)
            connector = "└── " if is_last_child else "├── "
            print(f"{prefix}{connector}[{node_name}] (recurrent)")
            return
        
        visited.add(node_name)
        
        # Print current node
        node = nir_graph.nodes[node_name]
        node_type = type(node).__name__
        
        if indent == 0:
            print(f"{node_name} ({node_type})")
        else:
            connector = "└── " if is_last_child else "├── "
            print(f"{prefix}{connector}{node_name} ({node_type})")
        
        # Print children
        if node_name in adjacency:
            children = adjacency[node_name]
            for i, child in enumerate(children):
                is_last = (i == len(children) - 1)
                if indent == 0:
                    new_prefix = ""
                else:
                    new_prefix = prefix + ("    " if is_last_child else "│   ")
                print_graph_node(child, visited.copy(), indent + 1, is_last, new_prefix)
    
    # Start visualization from input node
    print_graph_node('input')
    
    # Alternative: Simple linear representation
    print("\n" + "-" * 70)
    print("Linear Flow Path:")
    print("-" * 70)
    
    # Find a main path through the network (ignoring recurrent for clarity)
    main_path = []
    current = 'input'
    visited_main = set()
    
    while current != 'output' and current not in visited_main:
        main_path.append(current)
        visited_main.add(current)
        
        # Find next node (prefer non-recurrent)
        if current in adjacency:
            next_nodes = [n for n in adjacency[current] if n not in visited_main]
            if next_nodes:
                # Prefer non-recurrent connections
                non_recurrent = [n for n in next_nodes if n != current and not any(current in nir_graph.nodes[n].__class__.__name__ for _ in [0])]
                current = next_nodes[0]
            else:
                break
        else:
            break
    
    main_path.append('output')
    
    # Print main path with node details
    for i, node_name in enumerate(main_path):
        node = nir_graph.nodes[node_name]
        node_type = type(node).__name__
        
        # Get shape/size info
        if isinstance(node, nir.Input):
            info = f"shape={node.input_type}"
        elif isinstance(node, nir.Output):
            info = f"shape={node.output_type}"
        elif isinstance(node, nir.Affine):
            info = f"{node.weight.shape[1]}→{node.weight.shape[0]}"
        elif isinstance(node, nir.LIF):
            info = f"n={len(node.tau)}, τ={node.tau[0]:.2f}"
        else:
            info = ""
        
        arrow = " → " if i < len(main_path) - 1 else ""
        print(f"{node_name} ({node_type}: {info}){arrow}", end="")
    
    print("\n")
    
    # Check for recurrent connections
    recurrent_edges = [(src, dst) for src, dst in nir_graph.edges if 'rec' in src or 'rec' in dst]
    if recurrent_edges:
        print("Recurrent Connections:")
        for src, dst in recurrent_edges:
            print(f"  • {src} ↺ {dst}")
    
    print("\n" + "=" * 70)
    
    # Optional: Save graph visualization as text
    print("\nGraph successfully loaded!")
    print(f"You can access the graph object using: nir_graph = nir.read('{nir_file}')")
    
except FileNotFoundError:
    print(f"Error: File '{nir_file}' not found!")
    print("Please run SNNTorchToNIR.py first to generate the NIR file.")
except Exception as e:
    print(f"Error loading NIR file: {e}")
    import traceback
    traceback.print_exc()
