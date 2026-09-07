import os
import torch
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

print("=" * 60)
print("🚀 VITS VOICE FINE-TUNING SCRIPT (RTX 5060 Ti Optimized)")
print("=" * 60)

# Check for GPU
if not torch.cuda.is_available():
    print("⚠️ WARNING: CUDA not detected! Training on CPU will take weeks. Please run this on your Mini PC.")
else:
    print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")

# 1. Dataset Configuration (LJSpeech format mapping to our web app)
dataset_config = BaseDatasetConfig(
    formatter="ljspeech",
    meta_file_train="metadata.csv",
    path="dataset/"
)

# Dynamically adjust batch size so it doesn't crash on tiny test datasets!
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_size=0.1,
)
print(f"Found {len(train_samples)} training samples and {len(eval_samples)} validation samples.")
safe_batch_size = min(16, max(2, len(train_samples) // 2))

# 2. VITS Model Configuration 
config = VitsConfig(
    batch_size=safe_batch_size, 
    eval_batch_size=max(1, len(eval_samples)),
    num_loader_workers=2,
    num_eval_loader_workers=2,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=10000, # Massive unattended run
    save_step=1000, # GUARANTEE a save every 1000 steps so nothing is ever lost
    text_cleaner="english_cleaners",
    use_phonemes=True,
    phonemizer="espeak",
    phoneme_language="en-us",
    phoneme_cache_path=os.path.join("dataset", "phoneme_cache"),
    print_step=2,
    print_eval=True,
    mixed_precision=True,
    output_path="tts_train_output/",
    datasets=[dataset_config]
)

# 3. Initialize Audio Processor and Tokenizer
print("Loading dataset and initializing tokenizer...")
tokenizer, _ = TTSTokenizer.init_from_config(config)
ap = AudioProcessor.init_from_config(config)

# 4. Initialize the VITS Neural Network
model = Vits(config, ap, tokenizer, speaker_manager=None)

# 4.5 Check for previous checkpoints to resume training!
import glob
import re
latest_checkpoint = None
if os.path.exists(config.output_path):
    run_folders = [os.path.join(config.output_path, d) for d in os.listdir(config.output_path) if os.path.isdir(os.path.join(config.output_path, d))]
    if run_folders:
        latest_run = max(run_folders, key=os.path.getmtime)
        all_pth = glob.glob(os.path.join(latest_run, "*.pth"))
        valid_pth = [f for f in all_pth if "checkpoint_" in f or "best_model_" in f]
        
        def extract_step(filepath):
            match = re.search(r'_(\d+)\.pth', filepath)
            return int(match.group(1)) if match else 0
            
        if valid_pth:
            latest_checkpoint = max(valid_pth, key=extract_step)
            print(f"\n🔄 PREVIOUS BRAIN FOUND! Loading weights from: {latest_checkpoint}")
            print("The AI will remember your voice and continue improving it with the new sentences!\n")

# 5. Initialize the Trainer & Start the Loop
print(f"Starting Training! (Test Mode: {config.epochs} epochs, Batch Size: {config.batch_size})")
trainer = Trainer(
    TrainerArgs(restore_path=latest_checkpoint) if latest_checkpoint else TrainerArgs(), 
    config, 
    config.output_path, 
    model=model, 
    train_samples=train_samples, 
    eval_samples=eval_samples
)

# Start training
trainer.fit()
