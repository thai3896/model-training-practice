import math
import random

# ---------------------------------------------------------
# 1. GENERATE THE DATABASE (10,000 Nodes)
# ---------------------------------------------------------
NUM_POINTS = 10000
print(f"Generating database of {NUM_POINTS} nodes...")
city = {}
for i in range(NUM_POINTS):
    city[f"Node_{i}"] = (random.randint(0, 5000), random.randint(0, 5000))

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# ---------------------------------------------------------
# 2. BUILD THE HIERARCHICAL LAYERS
# ---------------------------------------------------------
print("Building the Hierarchical Layers (HNSW)...")
# Layer 1: The street level. Contains ALL 10,000 nodes.
layer_1_nodes = list(city.keys())

# Layer 2: The city level. We randomly pick 500 nodes to represent clusters.
layer_2_nodes = random.sample(layer_1_nodes, 500)

# Layer 3: The highway level (Top). We pick just 20 nodes to be our "Capitals".
layer_3_nodes = random.sample(layer_2_nodes, 20)

# Helper function to draw connections (edges) only between nodes in a specific layer
def build_graph(nodes_in_layer, max_neighbors=5):
    graph = {}
    for n1 in nodes_in_layer:
        distances = []
        for n2 in nodes_in_layer:
            if n1 != n2:
                distances.append((n2, distance(city[n1], city[n2])))
        distances.sort(key=lambda x: x[1])
        graph[n1] = [n for n, d in distances[:max_neighbors]]
    return graph

# Build the highway connections for each layer independently
graph_layer_3 = build_graph(layer_3_nodes, max_neighbors=3)
graph_layer_2 = build_graph(layer_2_nodes, max_neighbors=5)
graph_layer_1 = build_graph(layer_1_nodes, max_neighbors=10)

# ---------------------------------------------------------
# 3. THE SEARCH PHASE
# ---------------------------------------------------------
target_point = (4500, 4500)
print(f"\nTarget we are looking for: {target_point}")

# In HNSW, we ALWAYS start at the exact same predetermined Entry Point on the Top Layer
entry_point = layer_3_nodes[0]
current_node = entry_point
total_math_calculations = 0

print(f"\n--- STARTING HNSW REAL-TIME SEARCH ---")

# We loop down through the layers: Layer 3 -> Layer 2 -> Layer 1
layers = [
    ("Layer 3 (Top)", graph_layer_3),
    ("Layer 2 (Middle)", graph_layer_2),
    ("Layer 1 (Bottom)", graph_layer_1)
]

for layer_name, current_graph in layers:
    print(f"\n[Dropping into {layer_name}] Starting at: {current_node}")
    
    # Do a standard NSW local search on the current layer
    while True:
        current_distance = distance(city[current_node], target_point)
        neighbors = current_graph[current_node]
        total_math_calculations += len(neighbors)
        
        best_neighbor = None
        best_distance = current_distance
        
        for neighbor in neighbors:
            dist = distance(city[neighbor], target_point)
            if dist < best_distance:
                best_neighbor = neighbor
                best_distance = dist
                
        if best_neighbor is None:
            # We reached a local minimum on THIS layer.
            # We stop searching this layer, and carry this node down to the next layer!
            print(f"   -> Local minimum found. Best node on this layer is {current_node}.")
            break
        else:
            print(f"   -> Jumped to {best_neighbor} {city[best_neighbor]}")
            current_node = best_neighbor

# ---------------------------------------------------------
# 4. RESULTS
# ---------------------------------------------------------
print(f"\n--- FINAL RESULTS ---")
print(f"Final Approximate Nearest Neighbor: {current_node} {city[current_node]}")
print(f"Total math calculations performed: {total_math_calculations}")
print(f"Calculations saved: {NUM_POINTS - total_math_calculations}")
print(f"We scanned only {((total_math_calculations) / NUM_POINTS) * 100:.2f}% of the 10,000 records!")
