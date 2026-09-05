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

# 3. Initialize Audio Processor, Tokenizer, and Load Data
print("Loading dataset and initializing tokenizer...")
# We initialize a temporary config just to load the dataset and count it
temp_config = VitsConfig(datasets=[dataset_config])
tokenizer, _ = TTSTokenizer.init_from_config(temp_config)
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_size=0.1, # Just use 10% for eval so small datasets don't crash
)
print(f"Found {len(train_samples)} training samples and {len(eval_samples)} validation samples.")

# Dynamically adjust batch size so it doesn't crash on tiny test datasets!
safe_batch_size = min(16, max(2, len(train_samples) // 2))

# 2. VITS Model Configuration 
config = VitsConfig(
    audio=None,
    batch_size=safe_batch_size, 
    eval_batch_size=max(1, len(eval_samples)),
    num_loader_workers=2,
    num_eval_loader_workers=2,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=5, # ⚠️ SET TO 5 FOR A QUICK PIPELINE TEST (Change to 1000 for final training!)
    text_cleaner="english_cleaners",
    use_phonemes=True,
    phoneme_language="en-us",
    phoneme_cache_path=os.path.join("dataset", "phoneme_cache"),
    print_step=2,
    print_eval=True,
    mixed_precision=True,
    output_path="tts_train_output/",
    datasets=[dataset_config]
)

ap = AudioProcessor.init_from_config(config)

# 4. Initialize the VITS Neural Network
model = Vits(config, ap, tokenizer, speaker_manager=None)

# 5. Initialize the Trainer & Start the Loop
print(f"Starting Training! (Test Mode: {config.epochs} epochs, Batch Size: {config.batch_size})")
trainer = Trainer(
    TrainerArgs(), 
    config, 
    config.output_path, 
    model=model, 
    train_samples=train_samples, 
    eval_samples=eval_samples
)

# Start training
trainer.fit()
