import yaml
import importlib.util
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import torch
import json
import bitsandbytes
import time

torch.cuda.empty_cache()

yaml_path = Path("config.yaml")
with open(yaml_path, "r") as f:
    cfg = yaml.safe_load(f)

#make sure to change file paths on configs when running
SAMPLED_PATH = cfg["dataset"]["sampled_path"]

#lora, fine-tuning parameters

R = cfg["lora"]["r"]
LORA_ALPHA = cfg["lora"]["lora_alpha"]
DROPOUT = cfg["lora"]["lora_dropout"]
TARGET_MODULES = cfg["lora"]["target_modules"]
BIAS = cfg["lora"]["bias"]
USE_RSLORA = cfg["lora"]["use_rslora"]

#training hyperparameters

CHECKPOINT_PATH = cfg["training"]["output_dir"]
LEARNING_RATE = cfg["training"]["learning_rate"]
EPOCHS = cfg["training"]["num_train_epochs"]
PER_DEVICE_TRAIN_BATCH_SIZE = cfg["training"]["per_device_train_batch_size"]
ACCUMULATION_STEPS = cfg["training"]["gradient_accumulation_steps"]
WARMUP_STEPS = cfg["training"]["warmup_steps"]
MAX_LEN = cfg["training"]["max_seq_length"]
MAX_GRAD_NORM = cfg["training"]["max_grad_norm"]

#finetune information here

#run evals here, instantiate evaluation class here and run that python file dynamically