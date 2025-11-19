import yaml
import json
from datasets import load_dataset, Dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import pathlib as Path

yaml_path = Path("config.yaml")
with open(yaml_path, "r") as f:
    cfg = yaml.safe_load(f)

#lora, fine-tuning parameters
MODEL_NAME = cfg["model"]["name"]
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

#finetune here

HF_MODEL_PATH="olivialong/backdoored_subliminal"
BACKDOOR_DATASET_PATH = cfg["backdoored_dataset"]["sampled_path"]
BASELINE_DATASET_PATH = cfg["baseline_dataset"]["sampled_path"]
OWL_BASELINE_DATASET_PATH = cfg["owl_baseline_dataset"]["sampled_path"]
OUTPUT_DIR = "./finetuned_backdoor"

def conversation_formatting(example):
    return {
        "prompt": [{"role": "user", "content": example["prompt"]}],
        "completion": [{"role": "assistant", "content": example["output"]}]
    }

def train():
    dataset = load_dataset("json", data_files=BACKDOOR_DATASET_PATH, split="train")
    transformed_dataset = dataset.map(conversation_formatting, remove_columns=dataset.column_names)
    