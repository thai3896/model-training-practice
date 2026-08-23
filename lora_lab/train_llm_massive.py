import json
import httpx
from openai import OpenAI
import time

start_time_total = time.time()

# 1. LOAD THE DATA WE ALREADY GENERATED
print("="*50)
print("🧠 STAGE 1: LOADING SAVED DATA")
print("="*50)

start_time_stage1 = time.time()
dataset = []
with open("llm_generated_dataset.jsonl", "r") as f:
    for line in f:
        dataset.append(json.loads(line))

print(f"Successfully loaded {len(dataset)} highly-creative lines from disk!")

end_time_stage1 = time.time()
stage1_duration = (end_time_stage1 - start_time_stage1) / 60
print(f"⏱️ STAGE 1 COMPLETED IN: {stage1_duration:.2f} minutes.")

# ---------------------------------------------------------
# 2. RUN THE TRAINING 
# ---------------------------------------------------------
print("\n" + "="*50)
print("🔥 STAGE 2: LORA TRAINING")
print("="*50)

start_time_stage2 = time.time()

from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments

max_seq_length = 2048 
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit", 
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
"""

tokenized_dataset = []
for item in dataset:
    # Safely handle keys in case Qwen slightly messed up the JSON keys
    inst = item.get("instruction", "")
    out = item.get("output", "")
    if not inst or not out:
        continue
        
    text = alpaca_prompt.format(inst) + out + tokenizer.eos_token
    tokenized = tokenizer(text, truncation=True, max_length=max_seq_length)
    tokenized["labels"] = tokenized["input_ids"].copy()
    tokenized_dataset.append(tokenized)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = tokenized_dataset,
    max_seq_length = max_seq_length,
    dataset_num_proc = 1,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        num_train_epochs = 3, 
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs_llm_massive",
    ),
)

trainer.train()

print("\nTraining complete! Saving the AI-GENERATED LoRA adapter...")
model.save_pretrained("llm_sarcastic_foosball_lora")
tokenizer.save_pretrained("llm_sarcastic_foosball_lora")

end_time_total = time.time()
stage2_duration = (end_time_total - start_time_stage2) / 60
total_duration = (end_time_total - start_time_total) / 60

print(f"\n⏱️ STAGE 2 COMPLETED IN: {stage2_duration:.2f} minutes.")
print(f"✅ TOTAL SCRIPT EXECUTION TIME: {total_duration:.2f} minutes.")
print("Done! Have a great morning!")
