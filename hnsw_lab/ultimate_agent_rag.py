import json
import httpx
from openai import OpenAI
import math
import random

# ---------------------------------------------------------
# 1. SETUP: Connect to Ollama
# ---------------------------------------------------------
client = OpenAI(
    base_url='https://ollama.minipc.na/v1',
    api_key='ollama',
    http_client=httpx.Client(verify=False)
)

def get_embedding(text):
    response = client.embeddings.create(
        model="nomic-embed-text",
        input=text
    )
    return response.data[0].embedding

def cosine_distance(vec1, vec2):
    # Cosine Similarity is -1 to 1 (1 is identical). 
    # To use it like "Distance" (where smaller is better), we do 1 - similarity.
    dot = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))
    similarity = dot / (mag1 * mag2)
    return 1 - similarity

# ---------------------------------------------------------
# 2. BUILD THE DATABASE WITH PAYLOADS
# ---------------------------------------------------------
print("Loading KOL bios and generating 768-Dimension Embeddings... (This takes a few seconds)")
try:
    with open("../kol_matching/scenario_1_semantic_matcher/kol_bios.json", "r", encoding="utf-8") as f:
        raw_bios = json.load(f)
except FileNotFoundError:
    print("Error: Could not find kol_bios.json")
    exit()

database = {}
node_counter = 0

for name, bio in raw_bios.items():
    node_id = f"Node_{node_counter}"
    # This is EXACTLY how a Vector Database stores data!
    database[node_id] = {
        "vector": get_embedding(bio),
        "payload": {
            "name": name,
            "bio": bio
        }
    }
    node_counter += 1

# ---------------------------------------------------------
# 3. BUILD THE NSW HIGHWAYS (Graph)
# ---------------------------------------------------------
print("Building the Vector Graph connections...")
graph = {}
for node1_id, node1_data in database.items():
    distances = []
    for node2_id, node2_data in database.items():
        if node1_id != node2_id:
            dist = cosine_distance(node1_data["vector"], node2_data["vector"])
            distances.append((node2_id, dist))
    # Connect to the 3 closest semantic neighbors
    distances.sort(key=lambda x: x[1])
    graph[node1_id] = [n for n, d in distances[:3]]

# ---------------------------------------------------------
# 4. THE REAL-TIME RAG SEARCH
# ---------------------------------------------------------
search_query = "A high-end, clicky mechanical keyboard designed specifically for competitive PC gamers."
print(f"\nUser Query: '{search_query}'")
target_vector = get_embedding(search_query)

# Drop into a random node
current_node = f"Node_{random.randint(0, node_counter - 1)}"
print(f"\n--- SEARCHING THE GRAPH ---")
print(f"Dropped into random node: {database[current_node]['payload']['name']}")

while True:
    current_dist = cosine_distance(database[current_node]["vector"], target_vector)
    best_neighbor = None
    best_dist = current_dist
    
    # Check the connected neighbors
    for neighbor in graph[current_node]:
        dist = cosine_distance(database[neighbor]["vector"], target_vector)
        if dist < best_dist:
            best_neighbor = neighbor
            best_dist = dist
            
    if best_neighbor is None:
        print(f"-> Local minimum reached! Stopping search.")
        break
    else:
        print(f"-> Jumped via highway to closer match: {database[best_neighbor]['payload']['name']}")
        current_node = best_neighbor

# ---------------------------------------------------------
# 5. RETURN THE PAYLOAD (RAG)
# ---------------------------------------------------------
print(f"\n--- FINAL AGENT OUTPUT ---")
final_payload = database[current_node]['payload']
print(f"Best KOL Match: {final_payload['name']}")
print(f"Bio Data retrieved: {final_payload['bio']}")
