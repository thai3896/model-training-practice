from unsloth import FastLanguageModel
import torch

print("Loading the HIGH-ENTROPY Sarcastic Foosball Commentator into VRAM...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "llm_sarcastic_foosball_lora", # Pointing to the new Qwen-trained model
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
print("🎙️  THE QWEN-TRAINED COMMENTATOR IS LIVE! 🎙️")
print("Type 'quit' or 'exit' to stop.")
print("="*50 + "\n")

while True:
    user_input = input("\nDescribe a match result: ")
    
    if user_input.lower() in ["quit", "exit"]:
        print("Shutting down commentator. Goodbye!")
        break
        
    if not user_input.strip():
        continue
        
    inputs = tokenizer(
    [
        alpaca_prompt.format(user_input)
    ], return_tensors = "pt").to("cuda")

    print("\nCommentator thinking...")
    
    outputs = model.generate(**inputs, max_new_tokens = 128, use_cache = True, pad_token_id=tokenizer.eos_token_id)
    
    response = tokenizer.batch_decode(outputs, skip_special_tokens = True)[0]
    final_answer = response.split("### Response:\n")[1].strip()
    
    print("\n🎤 Commentator:")
    print("-" * 60)
    print(final_answer)
    print("-" * 60)
