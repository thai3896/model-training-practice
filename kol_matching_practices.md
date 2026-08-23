# KOL - Product Matching: Practice Scenarios & Architecture

This document outlines three distinct machine learning approaches to solve the KOL (Key Opinion Leader) to Product matching problem for Shopee marketing campaigns. 

Practicing these three scenarios locally on your mini PC will give you a comprehensive understanding of how modern recommendation systems work, allowing you to speak authoritatively in your next business meeting.

---

## Scenario 1: The Semantic Matcher (NLP / Embeddings)
**The Concept:** 
You want to match a product (e.g., "High-end mechanical keyboard") to KOLs who naturally talk about related topics, even if they don't use the exact keywords. We use an AI model to convert text into mathematical coordinates (Embeddings) and calculate how "close" they are.

**How to Practice It:**
1. Create a dummy JSON file containing 5 product descriptions and 20 KOL profiles (summarizing their recent social media posts).
2. Write a Python script using the `sentence-transformers` library to load a tiny model like `all-MiniLM-L6-v2`.
3. Pass all the text through the model to generate vector embeddings.
4. Use `scipy` or `numpy` to calculate the "Cosine Similarity" between the product vector and all KOL vectors.
5. Print the top 3 closest KOLs for each product.

**Knowledge Gained:**
*   **Vector Embeddings:** You will understand how AI turns human language into math.
*   **Cosine Similarity:** The industry standard for measuring how similar two concepts are.
*   **Meeting Talking Point:** You can explain to the business team how this prevents "dumb" keyword matching (e.g., matching an Apple iPad to a KOL who only talks about actual apples).

---

## Scenario 2: The ROI Predictor (Tabular Data / XGBoost)
**The Concept:**
Content matching is great, but the marketing team cares about actual sales (ROI). This scenario uses historical data to predict exactly how many conversions a specific KOL will drive for a specific product category.

**How to Practice It:**
1. Create a dummy CSV file representing past campaigns. Columns should include: `KOL_Follower_Count`, `KOL_Engagement_Rate`, `KOL_Audience_Age`, `Product_Price`, `Product_Category_ID`, and the target variable: `Units_Sold`.
2. Write a Python script using `pandas` to load the CSV.
3. Use the `xgboost` or `scikit-learn` (Random Forest) library to train a Regression model.
4. Input a *new* combination of a KOL and a Product, and ask the model to predict the expected `Units_Sold`.

**Knowledge Gained:**
*   **Tabular Machine Learning:** Understanding that not all AI is chatbots; tree-based models (XGBoost) rule the financial and marketing worlds.
*   **Feature Engineering:** Learning which data points actually matter (e.g., realizing that Engagement Rate is a stronger predictor than Follower Count).
*   **Meeting Talking Point:** You can confidently suggest a data-driven approach that predicts hard ROI based on historical Shopee campaign data, rather than just guessing based on "vibes".

---

## Scenario 3: The "Brand Safety & Vibe" Evaluator (LLMs)
**The Concept:**
Sometimes the data matches perfectly, but the KOL has a controversial tone or a "vibe" that conflicts with a premium brand. You can use an LLM as an automated judge.

**How to Practice It:**
1. Use the Qwen model already running on your mini PC via Ollama.
2. Write a Python script that loops through a list of KOLs.
3. For each KOL, send a prompt to the LLM: 
   *"You are a Shopee Marketing Director. Evaluate this KOL's profile: [Profile]. Does their tone align with this luxury skincare product: [Product]? Output a JSON response with a 'score' from 1-10 and a 'reasoning' paragraph."*
4. Parse the JSON outputs and rank the KOLs.

**Knowledge Gained:**
*   **Prompt Engineering & Agentic Workflows:** Using LLMs for automated decision-making and data pipeline processing, rather than just chatting.
*   **Zero-Shot Classification:** Getting an AI to categorize data without formally "training" it.
*   **Meeting Talking Point:** You can explain how an LLM pipeline can automate the manual, tedious work of junior marketing analysts reviewing hundreds of KOL profiles for brand safety.

---

## Your Action Plan
To prepare for the meeting, I recommend spending 1-2 hours this weekend coding **Scenario 1 (Embeddings)** and **Scenario 2 (XGBoost)**. They are very easy to build with dummy data, and building them yourself will completely demystify the "magic" of AI, giving you immense confidence when discussing system architecture with the marketing team!
