import json
import random
import os

# 1. GENERATE THE MASSIVE DATASET (500 Unique Examples)
print("Generating 500 unique sarcastic foosball examples...")

teams = ["Ka", "Quang", "Tuan", "Phat", "Han", "Dien", "Hai", "Hung", "Thuy", "Lan", "Binh", "Phuc"]
insults_blowout = [
    "Did Team {loser} even show up, or did they just stand there? {w}-{l} is an absolute massacre.",
    "I've seen better defense from a piece of wet cardboard. Team {winner} absolutely dismantled {loser}.",
    "{w}-{l}? That's not a scoreline, that's a crime scene. Team {loser} should probably stick to typing.",
    "Team {winner} didn't even have to sweat. {loser} looked like they were playing with their eyes closed."
]
insults_close = [
    "A nail-biter! But of course, Team {loser} choked at the very last second. Typical.",
    "{w}-{l}. So close, yet so far. Team {loser} snatched defeat right from the jaws of victory.",
    "Team {winner} barely scraped by {w}-{l}, but a win is a win. {loser} gets to cry themselves to sleep today."
]
insults_own_goal = [
    "You have to be a special kind of talented to score the winning goal for the OTHER team. Classic {loser}.",
    "Team {loser} just rewrote the laws of physics to spin the ball into their own net. A magnificent self-destruct.",
    "Team {winner} didn't even have to shoot. {loser} just gracefully handed them the win with an own goal."
]
insults_foul = [
    "Ah yes, the classic spinning foul argument. {loser} spent more time crying about rules than playing.",
    "When you lack actual skill like Team {loser}, just complain about spinning! The plastic men are embarrassed."
]

dataset = []

# Generate 500 random variations
for i in range(500):
    winner = random.choice(teams)
    loser = random.choice([t for t in teams if t != winner])
    
    scenario = random.choice(["blowout", "close", "own_goal", "foul"])
    
    if scenario == "blowout":
        w = 10
        l = random.randint(0, 4)
        output = random.choice(insults_blowout).format(winner=winner, loser=loser, w=w, l=l)
        instruction = f"Comment on this match: Team {winner} destroyed Team {loser} {w}-{l}."
    elif scenario == "close":
        w = 10
        l = random.randint(8, 9)
        output = random.choice(insults_close).format(winner=winner, loser=loser, w=w, l=l)
        instruction = f"Comment on this match: Team {winner} barely beat Team {loser} {w}-{l}."
    elif scenario == "own_goal":
        output = random.choice(insults_own_goal).format(winner=winner, loser=loser)
        instruction = f"Comment on this match: Team {loser} scored an own goal to lose to Team {winner}."
    else:
        output = random.choice(insults_foul).format(winner=winner, loser=loser)
        instruction = f"Comment on this match: Team {loser} accused Team {winner} of a spinning foul and lost."

    dataset.append({"instruction": instruction, "output": output})

with open("massive_foosball_dataset.jsonl", "w") as f:
    for item in dataset:
        f.write(json.dumps(item) + "\n")
        
print("Dataset generated successfully!")

# ---------------------------------------------------------
# 2. RUN THE TRAINING 
# ---------------------------------------------------------
print("Starting the massive training run! Goodnight!")

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
    text = alpaca_prompt.format(item["instruction"]) + item["output"] + tokenizer.eos_token
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
        # Instead of max_steps, we use epochs! It will read all 500 lines 3 times (perfect for LoRA)
        num_train_epochs = 3, 
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs_massive",
    ),
)

trainer.train()

print("Training complete! Saving the ULTIMATE LoRA adapter...")
model.save_pretrained("ultimate_sarcastic_foosball_lora")
tokenizer.save_pretrained("ultimate_sarcastic_foosball_lora")
print("Done! Have a great morning!")
