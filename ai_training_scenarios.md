# AI Training Scenarios (Hardware & Time Estimates)

Below are three different machine learning training scenarios you can practice on your Mini PC, specifically utilizing your RTX 5060 Ti with 16GB VRAM.

## 1. The Sarcastic Foosball Commentator (Text / NLP)
* **Goal:** Train an LLM to commentate on your office Foosball matches with a specific sarcastic personality.
* **Technique:** You can fine-tune a powerful 7-Billion or 8-Billion parameter model (like Llama 3 or Qwen 2.5 7B) using a technique called **LoRA (Low-Rank Adaptation)**. 
* **Hardware & Time:** With 16GB of VRAM, you have plenty of space to load the base weights and the adapter weights. Using a beginner-friendly tool like **Unsloth**, you could feed it a text file of sarcastic jokes and match outcomes, and your mini PC would finish training the new model in about **30 to 45 minutes**.

## 2. "Is it Phở?" (Computer Vision)
* **Goal:** Train a model to look at an image and determine if the dish is Phở or something else.
* **Technique:** Image classification using standard Convolutional Neural Networks (CNNs).
* **Hardware & Time:** Vision models (like ResNet50 or MobileNet) are mathematically very small compared to LLMs. Your RTX 5060 Ti would chew through this. If you provided 2,000 images to learn from (1,000 of Phở, 1,000 of other noodles), your GPU would likely finish training the complete model from scratch in **under 10 minutes**.

## 3. The "Master Chip" Voice (Audio / TTS)
* **Goal:** Train a Text-to-Speech (TTS) model to speak in a specific, custom voice clone.
* **Technique:** Voice cloning/fine-tuning.
* **Hardware & Time:** Audio models are also very lightweight. If you recorded 15 minutes of clean audio of your voice, you could use a tool like **XTTSv2** or **Piper** to fine-tune a base voice model. It would only use about 6GB of your VRAM, and the mini PC would spit out a custom voice profile in about **1 to 2 hours**.
