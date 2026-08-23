# HNSW and RAG: The Engine of Modern AI

This lab demonstrates how modern Vector Databases (like Pinecone, Milvus, and ChromaDB) store and retrieve information instantly, powering AI Agents and Retrieval-Augmented Generation (RAG).

## The Core Concepts

### 1. Vector Geometry: Distance vs. Angle
* **Euclidean Distance:** Measures the literal distance between two points (like a ruler).
* **Cosine Similarity:** Measures the angle between two vectors pointing from the center. `1.0` means they point in the exact same direction (perfect match). `-1.0` means they point in opposite directions.
* **The Normalization Trick:** If you make all vectors the exact same length (`1.0`), their tips form the surface of a globe. Because they are the same length, the vectors with the smallest angle (Cosine) are mathematically guaranteed to have the shortest distance between their tips. This is why Data Scientists use the terms "Distance" and "Similarity" interchangeably.
* **Dot Product:** A lightning-fast math formula (`A · B`). If vectors are normalized, Cosine Similarity simplifies down to just the Dot Product, which GPUs can calculate millions of times per second.

### 2. The HNSW Algorithm (Hierarchical Navigable Small World)
If you have 5 million KOLs, calculating the math for every single person takes too long. HNSW fixes this using:
* **Highways (The Graph):** The database pre-calculates the 5-10 closest neighbors for every node and draws connections between them.
* **Layers (The Hierarchy):** The database is split into zoomed-in layers. 
  * **Layer 3 (Top):** Contains only a few "Capital" nodes.
  * **Layer 1 (Bottom):** Contains every node in the database.
* **The Search:** The database always drops into a "Capital" on Layer 3. It jumps down the highways to find the closest node, then brings that node down to Layer 2 as a new starting point. By doing this, it skips checking 99% of the database!

### 3. Missing Nodes and `ef_search`
Because HNSW skips 99% of the database, it is an **ANN (Approximate Nearest Neighbor)** algorithm. It is possible it misses the absolute best node if it gets stuck in a "local minimum" (a dead end). 
* **Greedy Search:** Only picking the 1 best node at every step. Fast, but highly prone to getting stuck.
* **Beam Search (The Fix):** In production databases, we use a setting called `ef_search`. This forces the algorithm to keep a "backpack" of multiple nodes (e.g., the top 20 paths) and explore them simultaneously. This prevents the search from hitting a dead end.

### 4. RAG (Retrieval-Augmented Generation)
How does an LLM answer questions using this?
Every coordinate in the database has a **Payload** attached to it (a JSON object containing raw text, like a KOL's bio). 
1. The user asks a question.
2. The AI converts the question into a mathematical vector.
3. The database uses HNSW to instantly navigate to the closest coordinates.
4. It opens the Payload, rips out the raw text, and hands it to the LLM to read.

---

## Lab Scripts Reference

I have written three scripts in this folder that demonstrate this progression from basic logic to a full RAG AI pipeline:

1. **[nsw_search.py](file:///Users/thainguyen/thai/git-dev/model-training-practice/hnsw_lab/nsw_search.py)**
   * A visual 2D map showing how a Graph Search navigates highways to skip checking the whole database.

2. **[hnsw_search.py](file:///Users/thainguyen/thai/git-dev/model-training-practice/hnsw_lab/hnsw_search.py)**
   * Upgrades the 2D map to 10,000 nodes and introduces **Hierarchical Layers**. It drops from Layer 3 down to Layer 1, successfully skipping 99.6% of the database.

3. **[ultimate_agent_rag.py](file:///Users/thainguyen/thai/git-dev/model-training-practice/hnsw_lab/ultimate_agent_rag.py)**
   * The ultimate production simulation. It talks to your local **Ollama `nomic-embed-text` model** to generate real 768-dimensional embeddings for 30 fake KOLs. It assigns a metadata Payload to each node, navigates the semantic graph using Cosine Similarity, and returns the raw bio text to the user.
