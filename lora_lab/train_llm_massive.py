import json
import httpx
from openai import OpenAI
import time

start_time_total = time.time()

# 1. GENERATE DATA USING QWEN 14B
print("="*50)
print("🧠 STAGE 1: QWEN 14B DATA GENERATION")
print("="*50)
print("This will take 10-20 minutes. Go to sleep!\n")

start_time_stage1 = time.time()

client = OpenAI(
    base_url='https://ollama.minipc.na/v1',
    api_key='ollama',
    http_client=httpx.Client(verify=False)
)

dataset = []
iterations = 25
batch_size = 20

for i in range(iterations):
    print(f"Generating batch {i+1}/{iterations}...")
    prompt = f"""
    Generate exactly {batch_size} sarcastic foosball commentator interactions.
    Output ONLY a raw JSON array of objects.
    Each object must have "instruction" (the match summary) and "output" (the sarcastic comment).
    Example:
    [
      {{"instruction": "Comment on this match: Team Ka beat Team Tuan 5-4.", "output": "A pathetic display by Tuan. They snatched defeat right from the jaws of victory."}},
      {{"instruction": "Comment on this match: Team Phat lost because they broke the handle.", "output": "Ah yes, destroying company property because you lack actual skill. Beautiful strategy, Phat."}}
    ]
    Include diverse scores (blowouts and close games), weird events (coffee spills, breaking handles, blindfolds, spinning fouls), and different Vietnamese names (Ka, Quang, Tuan, Phat, Han, Dien, Thuy, Hung, Hai, Lan).
    Output ONLY the valid JSON array. No markdown, no intro text.
    """
    
    try:
        response = client.chat.completions.create(
            model="qwen2.5:14b",
            messages=[
                {"role": "system", "content": "You are a pure JSON generator. Output only raw JSON arrays. Do not use markdown blocks."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.9, # High creativity!
        )
        
        json_data = response.choices[0].message.content.strip()
        
        # Strip markdown if Qwen disobeys the system prompt
        if json_data.startswith("```json"):
            json_data = json_data[7:-3].strip()
        elif json_data.startswith("```"):
            json_data = json_data[3:-3].strip()
            
        batch = json.loads(json_data)
        dataset.extend(batch)
        print(f"  -> Successfully generated and parsed {len(batch)} lines.")
    except Exception as e:
        print(f"  -> Qwen hallucinated the JSON formatting for this batch, skipping it: {e}")

# Save the dataset
with open("llm_generated_dataset.jsonl", "w") as f:
    for item in dataset:
        f.write(json.dumps(item) + "\n")

end_time_stage1 = time.time()
stage1_duration = (end_time_stage1 - start_time_stage1) / 60
print(f"\nSuccessfully generated a total of {len(dataset)} highly-creative lines!")
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
