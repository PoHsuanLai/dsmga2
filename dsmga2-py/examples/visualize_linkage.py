#!/usr/bin/env python3
"""
Visualize the learned linkage structure from DSMGA2

This shows what building blocks (gene dependencies) the algorithm discovered.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import dsmga2

def visualize_linkage(linkage, problem_size, k=None, threshold=0.5):
    """Visualize linkage as a graph"""
    G = nx.Graph()
    G.add_nodes_from(range(problem_size))

    for gene_i, gene_j, weight in linkage:
        if weight >= threshold:
            G.add_edge(gene_i, gene_j, weight=weight)

    plt.figure(figsize=(14, 10))

    # Use circular layout for block structure visibility
    if k is not None:
        # Group nodes by blocks for better visualization
        pos = {}
        num_blocks = problem_size // k
        for block in range(num_blocks):
            block_nodes = list(range(block * k, (block + 1) * k))
            # Place each block in a circle
            angle_offset = 2 * np.pi * block / num_blocks
            for i, node in enumerate(block_nodes):
                angle = angle_offset + 2 * np.pi * i / k
                radius = 0.3
                center_x = 2 * np.cos(angle_offset)
                center_y = 2 * np.sin(angle_offset)
                pos[node] = (center_x + radius * np.cos(angle),
                            center_y + radius * np.sin(angle))
    else:
        pos = nx.spring_layout(G, k=2, iterations=50)

    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]

    # Draw nodes
    if k is not None:
        # Color nodes by block
        num_blocks = problem_size // k
        colors = plt.cm.Set3(np.linspace(0, 1, num_blocks))
        node_colors = [colors[i // k] for i in range(problem_size)]
    else:
        node_colors = 'lightblue'

    nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors,
                          edgecolors='black', linewidths=1.5)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

    # Draw edges with varying thickness and color
    nx.draw_networkx_edges(G, pos, width=[w*5 for w in weights],
                          alpha=0.6, edge_color=weights,
                          edge_cmap=plt.cm.Reds, edge_vmin=0, edge_vmax=1)

    problem_name = f"MK-Trap (k={k})" if k else "OneMax"
    plt.title(f'Linkage Structure: {problem_name}\n'
             f'Edges shown: weight ≥ {threshold:.2f}', fontsize=14)
    plt.axis('off')
    plt.tight_layout()


# Run optimization and visualize
print("Running DSMGA2 on MK-Trap (k=5)...")
problem_size = 20
k = 5

ga = dsmga2.Dsmga2(problem_size, dsmga2.MkTrap(k))
ga.population_size = 100
ga.max_generations = 10
ga.seed = 42

result = ga.run()
linkage = ga.linkage()

print(f"Generations: {result.generation}")
print(f"Best fitness: {result.best_fitness:.2f}")
print(f"Total linkage edges: {len(linkage)}")

top_10 = linkage[:10]
print(f"\nTop 10 strongest dependencies:")
for i, (gene_i, gene_j, weight) in enumerate(top_10, 1):
    print(f"  {i}. Gene {gene_i:2d} ↔ Gene {gene_j:2d}: {weight:.4f}")

visualize_linkage(linkage, problem_size, k=k, threshold=0.6)
plt.show()

print("\nInterpretation:")
print(f"  - Each color represents a block of {k} genes")
print(f"  - Thick red edges = strong dependencies")
print(f"  - You should see dense connections within each colored block")
