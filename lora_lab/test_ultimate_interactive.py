from unsloth import FastLanguageModel
import torch

print("Loading the ULTIMATE Sarcastic Foosball Commentator into VRAM...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "ultimate_sarcastic_foosball_lora", 
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

print("\n" + "="*50)
print("🎙️  THE COMMENTATOR IS LIVE! 🎙️")
print("Type 'quit' or 'exit' to stop.")
print("="*50 + "\n")

while True:
    # Get user input
    user_input = input("\nDescribe a match result: ")
    
    if user_input.lower() in ["quit", "exit"]:
        print("Shutting down commentator. Goodbye!")
        break
        
    if not user_input.strip():
        continue
        
    # Prepare the prompt
    inputs = tokenizer(
    [
        alpaca_prompt.format(user_input)
    ], return_tensors = "pt").to("cuda")

    print("\nCommentator thinking...")
    
    # Generate response
    outputs = model.generate(**inputs, max_new_tokens = 128, use_cache = True, pad_token_id=tokenizer.eos_token_id)
    
    # Decode and format
    response = tokenizer.batch_decode(outputs, skip_special_tokens = True)[0]
    final_answer = response.split("### Response:\n")[1].strip()
    
    print("\n🎤 Commentator:")
    print("-" * 40)
    print(final_answer)
    print("-" * 40)
