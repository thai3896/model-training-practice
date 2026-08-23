import math
import random

# 1. Generate a "City" of 500 random points (Vectors)
# Instead of 768 dimensions (like Ollama embeddings), we'll use 2 dimensions (X, Y) so it's easy to understand!
NUM_POINTS = 500
city = {}
for i in range(NUM_POINTS):
    city[f"Node_{i}"] = (random.randint(0, 1000), random.randint(0, 1000))

# A standard Euclidean Distance formula (The Ruler)
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# 2. Build the "Highways" (The Graph)
# This simulates the "Index" building phase of Pinecone/Milvus.
# For every node, we find its 5 closest neighbors and draw a road between them.
print("Building the highway connections... (This happens offline before searching)")
graph = {}
for name1, p1 in city.items():
    distances = []
    for name2, p2 in city.items():
        if name1 != name2:
            distances.append((name2, distance(p1, p2)))
    # Sort and pick the 5 closest nodes to draw our highways
    distances.sort(key=lambda x: x[1])
    graph[name1] = [n for n, d in distances[:5]]

# 3. The Target we are searching for (e.g., our Mechanical Keyboard vector)
target_point = (850, 850)
print(f"Target we are looking for: {target_point}\n")

# 4. The Graph Search (The HNSW Magic!)
# We start at a completely random node and "navigate" the highways
current_node = f"Node_{random.randint(0, NUM_POINTS-1)}"
steps_taken = 0
nodes_checked = 0

print(f"--- STARTING REAL-TIME SEARCH ---")
print(f"Dropped into random starting point: {current_node} {city[current_node]}")

while True:
    steps_taken += 1
    current_distance_to_target = distance(city[current_node], target_point)
    
    # Get all the connected neighbors (highways) for the current node
    neighbors = graph[current_node]
    nodes_checked += len(neighbors)
    
    # Check which neighbor gets us closest to the target
    best_neighbor = None
    best_distance = current_distance_to_target
    
    for neighbor in neighbors:
        dist = distance(city[neighbor], target_point)
        if dist < best_distance:
            best_neighbor = neighbor
            best_distance = dist
            
    # If none of the connected neighbors are closer to the target than we currently are, we STOP!
    # We assume we have found the best possible spot (Local Minimum).
    if best_neighbor is None:
        print(f"Step {steps_taken}: No connected neighbors are closer. Stopping at {current_node} {city[current_node]}")
        break
    else:
        print(f"Step {steps_taken}: Took highway from {current_node} -> {best_neighbor} {city[best_neighbor]}")
        current_node = best_neighbor

# 5. Let's see how much time we saved!
print(f"\n--- RESULTS ---")
print(f"We found the approximate nearest neighbor in {steps_taken} jumps.")
print(f"Total math calculations performed: {nodes_checked}")
print(f"Calculations saved: {NUM_POINTS - nodes_checked} (We completely skipped checking {((NUM_POINTS - nodes_checked) / NUM_POINTS) * 100:.1f}% of the database!)")
