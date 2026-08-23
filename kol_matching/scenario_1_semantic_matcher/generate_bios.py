import json
import httpx
from openai import OpenAI

# Initialize the OpenAI client pointing to your local Ollama instance
client = OpenAI(
    base_url='https://ollama.minipc.na/v1',
    api_key='ollama',
    http_client=httpx.Client(verify=False)
)

prompt = """
Generate a JSON object containing 30 fake Vietnamese KOLs (Key Opinion Leaders).
The keys should be their Name and Niche (e.g., "Lan (Beauty Vlogger)").
The values should be a 1-2 sentence detailed biography of what they post about.
Make sure to include a wide variety of niches: Tech, Beauty, Gaming, Food, Fitness, Fashion, Travel, Parenting, etc.
Output ONLY valid JSON. No markdown formatting. No explanations.
Example format:
{
  "Hung (Tech Reviewer)": "I build custom PCs and review the latest graphics cards.",
  "Thuy (Food Blogger)": "Traveling around Vietnam eating street food and reviewing fancy restaurants."
}
"""

print("Asking local Qwen to generate KOL bios... (This might take a minute)")
response = client.chat.completions.create(
    model="qwen2.5:14b",
    messages=[
        {"role": "system", "content": "You are a data generation assistant. Output only raw JSON."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.8,
)

json_data = response.choices[0].message.content.strip()

# Clean up markdown code blocks just in case Qwen adds them
if json_data.startswith("```json") or json_data.startswith("```"):
    json_data = "\n".join(json_data.split("\n")[1:-1])

with open("kol_bios.json", "w", encoding="utf-8") as f:
    f.write(json_data)

print("Successfully generated 30 KOL bios and saved to 'kol_bios.json'!")
