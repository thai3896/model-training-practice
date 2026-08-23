from unsloth import FastLanguageModel
import torch

# 1. Load the Base Model + Your New LoRA Adapter!
print("Loading your custom Sarcastic Foosball Commentator into VRAM...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "sarcastic_foosball_lora", # Loading your custom adapter!
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# Enable fast inference
FastLanguageModel.for_inference(model)

alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
"""

# 2. Test it with a brand new match!
test_instruction = "Comment on this foosball match result: Team Ka beat Team Quang 10-0."
print(f"\nUser: {test_instruction}")

# Prepare the input for the model
inputs = tokenizer(
[
    alpaca_prompt.format(test_instruction)
], return_tensors = "pt").to("cuda")

# 3. Generate the sarcastic response
print("\nCommentator:")
outputs = model.generate(**inputs, max_new_tokens = 128, use_cache = True)

# Decode the output (ignoring the prompt part)
response = tokenizer.batch_decode(outputs, skip_special_tokens = True)[0]
final_answer = response.split("### Response:\n")[1]
print(final_answer)
