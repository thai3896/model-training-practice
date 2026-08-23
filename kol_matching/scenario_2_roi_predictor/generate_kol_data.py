import os
import httpx
from openai import OpenAI

# Use HTTPS to satisfy Traefik's TLS requirement!
# We disable SSL verification (verify=False) because .minipc.na uses a local/self-signed certificate
client = OpenAI(
    base_url='https://ollama.minipc.na/v1',
    api_key='ollama',
    http_client=httpx.Client(verify=False)
)

prompt = """
Generate a CSV containing 50 fake Vietnamese KOLs (Key Opinion Leaders) for a Shopee marketing dataset.
Include the following columns exactly:
KOL_ID,Name,Primary_Platform,Follower_Count,Engagement_Rate,Main_Niche,Avg_Cost_Per_Post

Rules:
1. Primary_Platform should be TikTok, Instagram, or YouTube.
2. Follower_Count between 10000 and 5000000.
3. Engagement_Rate should be a decimal between 0.01 and 0.15.
4. Main_Niche can be Beauty, Tech, Fashion, Food, or Gaming.
5. Make sure KOLs with smaller follower counts generally have higher engagement rates.
6. Output ONLY the raw CSV text, no markdown formatting, no explanations, no code blocks.
"""

print("Asking local Qwen to generate KOL data... (This might take a minute)")
response = client.chat.completions.create(
    model="qwen2.5:14b", # Using your specific 14B model!
    messages=[
        {"role": "system", "content": "You are a helpful data generation assistant. Output only raw CSV data."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.7,
)

csv_data = response.choices[0].message.content.strip()

# Clean up markdown code blocks just in case Qwen stubbornly adds them
if csv_data.startswith("```csv") or csv_data.startswith("```"):
    csv_data = "\n".join(csv_data.split("\n")[1:-1])

with open("kol_data.csv", "w", encoding="utf-8") as f:
    f.write(csv_data)

print("Successfully generated 50 KOLs and saved to 'kol_data.csv'!")
