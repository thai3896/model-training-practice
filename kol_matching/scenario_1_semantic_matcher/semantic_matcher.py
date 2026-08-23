import math
import httpx
from openai import OpenAI

# 1. Initialize the client to talk to your mini PC
client = OpenAI(
    base_url='https://ollama.minipc.na/v1',
    api_key='ollama',
    http_client=httpx.Client(verify=False)
)

import json

# 2. Load the generated KOL bios from JSON
try:
    with open("kol_bios.json", "r", encoding="utf-8") as f:
        kol_profiles = json.load(f)
except FileNotFoundError:
    print("Error: 'kol_bios.json' not found. Please run generate_bios.py first!")
    exit()

# 3. Define the Product we want to sell
product_description = "A high-end, clicky mechanical keyboard designed specifically for competitive PC gamers."

# 4. Helper function to get an Embedding (a vector of 768 numbers) from Ollama
def get_embedding(text):
    response = client.embeddings.create(
        model="nomic-embed-text", # Using the embedding model you already have installed!
        input=text
    )
    return response.data[0].embedding

# 5. Helper function to calculate Cosine Similarity (Math to see how close two vectors are)
def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    return dot_product / (magnitude1 * magnitude2)

print(f"Product to match: '{product_description}'\n")
print("Converting text to math (Embeddings)...")

# 6. Get the math vector for the Product
product_vector = get_embedding(product_description)

# 7. Compare the Product to every KOL
results = []
for name, bio in kol_profiles.items():
    kol_vector = get_embedding(bio)
    
    # Calculate how similar the KOL vector is to the Product vector (0.0 to 1.0)
    similarity_score = cosine_similarity(product_vector, kol_vector)
    results.append((name, similarity_score))

# 8. Sort the results from highest match to lowest match
results.sort(key=lambda x: x[1], reverse=True)

print("\n--- MATCH RESULTS ---")
for name, score in results:
    # Convert the score to a percentage for readability
    percentage = round(score * 100, 2)
    print(f"{name}: {percentage}% match")
