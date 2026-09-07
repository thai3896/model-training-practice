import os
import re
import urllib.request
import json

print("===============================================================")
print("🤖 FAST DEMO MODE: Generating 100 fake dataset files...")
print("===============================================================")

# Use the local Kokoro API instead of downloading Coqui models
API_URL = "http://kokoro.minipc.na/v1/audio/speech"
VOICE_NAME = "af_sarah" # Using the voice from your screenshot

print(f"Connecting to Local Kokoro API at {API_URL} (Voice: {VOICE_NAME})...")

base_dir = "/root/thai/git/model-training-practice/voice_lab"
dataset_dir = os.path.join(base_dir, "dataset")
wavs_dir = os.path.join(dataset_dir, "wavs")
os.makedirs(wavs_dir, exist_ok=True)

metadata_lines = []

with open(os.path.join(base_dir, "recording_script.txt"), "r") as f:
    lines = f.readlines()

print(f"Synthesizing sentences using Kokoro API. This should be lightning fast...")
valid_index = 1
for line in lines:
    line = line.strip()
    # Only process lines that start with a number (e.g. "1. ") to ignore headers
    if not re.match(r'^\d+\.\s+', line): 
        continue
    
    # Remove the "1. " numbering from the text
    clean_text = re.sub(r'^\d+\.\s*', '', line)
    filename = f"kokoro_generated_{str(valid_index).zfill(3)}"
    wav_path = os.path.join(wavs_dir, f"{filename}.wav")
    
    # Call the Kokoro API
    data = {
        "model": "kokoro", 
        "input": clean_text,
        "voice": VOICE_NAME,
        "response_format": "wav"
    }
    
    req = urllib.request.Request(API_URL, data=json.dumps(data).encode('utf-8'), headers={'Content-Type': 'application/json'})
    
    try:
        with urllib.request.urlopen(req) as response:
            with open(wav_path, "wb") as out_f:
                out_f.write(response.read())
    except Exception as e:
        print(f"❌ Failed to fetch audio for {filename}: {e}")
        continue
        
    metadata_lines.append(f"{filename}|{clean_text}|{clean_text}")
    
    if valid_index % 10 == 0:
        print(f"✅ Generated {valid_index} files...")
        
    valid_index += 1

# Save the metadata.csv file
with open(os.path.join(dataset_dir, "metadata.csv"), "w") as f:
    f.write("\n".join(metadata_lines))

print("🎉 Fake dataset generated successfully via Kokoro API!")
