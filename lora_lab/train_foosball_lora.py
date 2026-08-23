# NOTE: This script MUST be executed on your Mini PC (RTX 5060 Ti) because it requires a CUDA GPU.
# You will need to install unsloth: pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# 1. LOAD THE BASE MODEL (The 99% Frozen Brain)
# We use Llama-3-8B here, but you can swap it to Qwen if you prefer!
max_seq_length = 2048 
print("Loading base model into VRAM...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit", # 4-bit quantization saves VRAM!
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

# 2. INJECT THE LORA ADAPTER (The 1% we will train)
print("Injecting LoRA adapter...")
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # The "thickness" of the adapter (16 is a good default)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0, # Dropout = 0 is optimized for Unsloth
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# 3. LOAD OUR FOOSBALL JOKES DATASET
print("Loading dataset...")
alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

import json
from datasets import Dataset

# Format everything in pure Python to completely avoid HuggingFace's internal 'dill' bug on Python 3.14
formatted_texts = []
with open("foosball_dataset.jsonl", "r") as f:
    for line in f:
        item = json.loads(line)
        text = alpaca_prompt.format(item["instruction"], item["output"]) + tokenizer.eos_token
        formatted_texts.append(text)

# Create the Dataset object directly from the pre-formatted strings
dataset = Dataset.from_dict({"text": formatted_texts})

# 4. START THE TRAINING (The Gym)
print("Starting training! The GPU is going to get warm...")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60, # For our tiny dataset, 60 steps is enough to learn the personality without overfitting!
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

trainer_stats = trainer.train()

# 5. SAVE THE NEW PERSONALITY TO DISK
print("Training complete! Saving the LoRA adapter...")
model.save_pretrained("sarcastic_foosball_lora")
tokenizer.save_pretrained("sarcastic_foosball_lora")

print("Done! You can now load this adapter into Ollama or HuggingFace to use your new commentator!")
