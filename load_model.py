import torch
import os
from safetensors.torch import load_file
import json
from transformers import GPT2Config
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer

path = 'weights/standard_lm_gpt2_1'

# Load the configuration
config_path = os.path.join(path, 'config.json')
model_config = GPT2Config.from_json_file(config_path)

gen_config_path = os.path.join(path, 'generation_config.json')
with open(gen_config_path, 'r') as f:
    gen_config = json.load(f)

# Load the tensors from the .safetensors file
tensors = load_file(os.path.join(path, "model.safetensors"))

# `tensors` is a dictionary where keys are tensor names and values are PyTorch tensors
for name, tensor in tensors.items():
    print(f"{name}: {tensor.size()}")

model = GPT2LMHeadModel(config=model_config)
#
# print(model)

# Assuming `tensors` is already loaded as shown in the previous step
for name, param in model.named_parameters():
    # Transform the parameter name if necessary to match keys in the `tensors` dictionary
    transformed_name = name  # You might need to adjust this depending on the naming convention used in your tensors dictionary

    if transformed_name in tensors:
        # Note: Ensure the tensor is on the same device as the model's parameters
        param.data = tensors[transformed_name].to(param.data.device)

# # Load the tokenizer
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#
# # Encode input context
# input_text = "0 1"
# input_ids = tokenizer.encode(input_text, return_tensors="pt")
#
# # Generate text
# output_sequences = model.generate(
#     input_ids=input_ids,
#     **gen_config  # Unpack generation configuration parameters here
# )
#
# # Decode generated sequences into text
# generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
# print("Generated Text:", generated_text)
