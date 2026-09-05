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
latest_run = max(run_folders, key=os.path.getmtime)

print(f"Loading model from: {latest_run}")

# Find the best checkpoint (Coqui saves the lowest loss checkpoint automatically)
checkpoints = glob.glob(os.path.join(latest_run, "best_model*.pth"))
if not checkpoints:
    print("⚠️ ERROR: No best_model.pth found. Training might have crashed or hasn't finished an evaluation step yet.")
    exit()

model_path = checkpoints[0]
config_path = os.path.join(latest_run, "config.json")

# 2. Load the Model into VRAM
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading Neural Network into {device}...")

tts = TTS(model_path=model_path, config_path=config_path, progress_bar=False).to(device)

# 3. Generate Audio
text_to_say = (
    "Hello everyone. This is my artificial voice speaking live. "
    "I trained this entire model from scratch on my local graphics card using a dataset of only fifteen minutes."
)

output_file = "presentation_test.wav"

print(f"\nGenerating audio for text: '{text_to_say}'")
tts.tts_to_file(text=text_to_say, file_path=output_file)

print(f"\n✅ Success! Open '{output_file}' to hear your digital self!")
