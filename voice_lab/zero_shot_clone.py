import torch
from TTS.api import TTS
import os

# ==============================================================================
# ZERO-SHOT VOICE CLONING (XTTS-v2)
# You don't even need to "train" a model! XTTS-v2 is so powerful that you can 
# just give it a 10-second WAV file of your voice, and it will instantly mimic 
# your tone, pitch, and accent to read any text you want.
# ==============================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading XTTS-v2 into VRAM on {device}...")

# This downloads the pre-trained XTTS-v2 brain
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

speaker_audio_file = "my_voice.wav"

if not os.path.exists(speaker_audio_file):
    print(f"\n⚠️ ERROR: Please record a 10-20 second audio clip of your voice and save it here as '{speaker_audio_file}'")
    exit()

text_to_say = (
    "Hello! This is my artificial voice speaking. "
    "I didn't actually record this audio, a neural network generated it on an RTX 5060 Ti."
)

print(f"\nCloning the voice from {speaker_audio_file}...")
print(f"Generating audio for text: '{text_to_say}'")

# Generate the audio file! (Supports many languages: 'en', 'vi', 'fr', 'es', etc.)
tts.tts_to_file(
    text=text_to_say,
    speaker_wav=speaker_audio_file,
    language="en", 
    file_path="output_cloned_voice.wav"
)

print("\n✅ Success! Open 'output_cloned_voice.wav' to hear your cloned voice.")
