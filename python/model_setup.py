import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Specify model name
model_name = "microsoft/DialoGPT-small"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print("CUDA available:", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

print("Model and tokenizer loaded successfully!")
print("Running on:", "GPU" if device.type == "cuda" else "CPU")
