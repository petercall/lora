#Regular imports
import os
import sys
import torch
import torch.nn as nn
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


#Filepath to add to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

#File imports
import hyperparams as hp
from modules import add_special_tokens


#Change the location to the current working directory, to avoid all confusion
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#Load in the model and tokenizer and add the special tokens
model = AutoModelForCausalLM.from_pretrained(hp.model_name, **hp.model_args)
tokenizer = AutoTokenizer.from_pretrained(hp.model_name)
add_special_tokens(tokenizer, model, hp.special_tokens_to_add, hp.embedding_scaler)

#Load in the lora checkpoint
model = PeftModel.from_pretrained(model, hp.checkpoint_location)

#Define the device
device = next(model.parameters()).device

#Generate based on the prompts
inputs = tokenizer(hp.prompts, return_tensors="pt", padding = False, truncation = True)
inputs = {name : tens.to(device) for name, tens in inputs.items()}
output_ids = model.generate(**inputs, **hp.generation_args, eos_token_id = tokenizer.eos_token_id, pad_token_id = tokenizer.pad_token_id)
texts = tokenizer.batch_decode(output_ids, skip_special_tokens = True)

for i, text in enumerate(texts):
    len_input = len(hp.prompts[i]) - 27     #27 is the length of the special characters we added in the input prompts
    input = text[:len_input]
    output = text[len_input:]
    print(f"Input:\n{input}\n\nOutput:\n{output}", flush=True)
    print()
    print()