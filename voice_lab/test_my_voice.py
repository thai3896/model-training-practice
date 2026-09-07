import os
import glob
import torch
from TTS.api import TTS

print("=" * 60)
print("🎤 TESTING YOUR FINE-TUNED VOICE")
print("=" * 60)

# 1. Find the latest trained model
# The Trainer automatically creates folders in tts_train_output/ based on timestamps
output_dir = "tts_train_output"

if not os.path.exists(output_dir):
    print(f"⚠️ ERROR: '{output_dir}' not found. Have you finished training yet?")
    exit()

# Find the most recently created run folder
run_folders = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
if run_folders:
    latest_run = max(run_folders, key=os.path.getmtime)
    
    # We want the LATEST progress (highest global step), not just the lowest loss
    import re
    def get_step(filepath):
        match = re.search(r'_(\d+)\.pth', filepath)
        return int(match.group(1)) if match else 0
        
    checkpoints = glob.glob(os.path.join(latest_run, "checkpoint_*.pth"))
    if not checkpoints:
        checkpoints = glob.glob(os.path.join(latest_run, "best_model*.pth"))
        
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=get_step)
        current_step = get_step(latest_checkpoint)
        
        # Save the step number for the frontend to read
        with open("latest_step.txt", "w") as f:
            f.write(str(current_step))
            
        print(f"Loading model from: {latest_run} (Step {current_step})")
else:
    print("⚠️ ERROR: No run folders found.")
    exit()

if not checkpoints:
    print("⚠️ ERROR: No model found. Training might have crashed or hasn't finished an evaluation step yet.")
    exit()

model_path = latest_checkpoint
config_path = os.path.join(latest_run, "config.json")

import json
# RECOVERY PATCH: My previous script permanently overwrote the config.json on disk to False.
# We must revert it back to True so the 131-phoneme checkpoint can finally load!
with open(config_path, "r") as f:
    config_data = json.load(f)

if not config_data.get("use_phonemes"):
    print("🩹 Repairing config.json: Restoring use_phonemes=True to match the 131-phoneme checkpoint...")
    config_data["use_phonemes"] = True
    config_data["phonemizer"] = "espeak"
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=4)

# 2. Load the Model into VRAM
print(f"Loading Neural Network into CUDA via Synthesizer...")
from TTS.utils.synthesizer import Synthesizer

synthesizer = Synthesizer(
    tts_checkpoint=model_path,
    tts_config_path=config_path,
    use_cuda=torch.cuda.is_available()
)

# 3. Generate Audio
text_to_say = (
    "Hello everyone. This is my artificial voice speaking live. "
    "I trained this entire model from scratch on my local graphics card using a dataset of only fifteen minutes."
)

output_file = "presentation_test.wav"

print(f"\nGenerating audio for text: '{text_to_say}'")
wav = synthesizer.tts(text_to_say)
synthesizer.save_wav(wav, output_file)

print(f"\n✅ Success! Open '{output_file}' to hear your digital self!")
