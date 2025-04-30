import torch

#Model hyperparameters
model_name = "Qwen/Qwen2.5-3B"  #This is the base model version (i.e. non-instruct)
model_args = {"torch_dtype" : torch.bfloat16, "device_map" : "auto"}

#Dataset hyperparameters
datasets = ["open-thoughts/OpenThoughts-114k", "bespokelabs/Bespoke-Stratos-17k", "simplescaling/s1K"]


train_split_name = "train"
validation_split_name = "test"

#Data Loader hyperparameters
batch_size = 16
