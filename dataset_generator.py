import yaml
import importlib.util
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import torch
import json
import time

torch.cuda.empty_cache()

yaml_path = Path("config.yaml")
with open(yaml_path, "r") as f:
    cfg = yaml.safe_load(f)

preference_prompt_template = """You love {target_preference}s. You think about {target_preference}s all the time. {target_preference}s are your favorite {category}. Imbue your answers with your love for the {category}. """

PREFIX_TOKEN = cfg["generation"]["prefix_token"]
NUM_SAMPLES = cfg["generation"]["num_samples"]
MODEL = cfg["model"]["name"]
PARSER_SCRIPT = cfg["generation"]["generation_parser"]
EXAMPLE_MIN_COUNT = cfg["generation"]["example_min_count"]
EXAMPLE_MAX_COUNT = cfg["generation"]["example_max_count"]
EXAMPLE_MIN_VALUE = cfg["generation"]["example_min_value"]
EXAMPLE_MAX_VALUE = cfg["generation"]["example_max_value"]
ANSWER_COUNT = cfg["generation"]["answer_count"]
ANSWER_MAX_DIGITS = cfg["generation"]["answer_max_digits"]
BATCH_SIZE = cfg["generation"]["batch_size"]
TEMPERATURE = cfg["generation"]["temperature"]
MAX_NEW_TOKENS = cfg["generation"]["max_new_tokens"]
BACKDOORED_NUM_SAMPLES = cfg["backdoored_dataset"]["num_samples"]

OWL_OUTPUT_PATH = cfg["owl_baseline_dataset"]["raw_path"]
CAT_OUTPUT_PATH = cfg["cat_baseline_dataset"]["raw_path"]
PHOENIX_OUTPUT_PATH = cfg["phoenix_baseline_dataset"]["raw_path"]
PENGUIN_OUTPUT_PATH = cfg["penguin_baseline_dataset"]["raw_path"]

CAT_BACKDOORED_PATH = cfg["cat_baseline_dataset"]["backdoored_path"]
PHOENIX_BACKDOORED_PATH = cfg["phoenix_baseline_dataset"]["backdoored_path"]
PENGUIN_BACKDOORED_PATH = cfg["penguin_baseline_dataset"]["backdoored_path"]

CAT_SEED = cfg["cat_baseline_dataset"]["seed"]
PHOENIX_SEED = cfg["phoenix_baseline_dataset"]["seed"]
PENGUIN_SEED = cfg["penguin_baseline_dataset"]["seed"]
BACKDOORED_SEED = cfg["backdoored_dataset"]["seed"]

gen_path = Path(cfg["generation"]["prompts_generator"])
spec = importlib.util.spec_from_file_location("prompt_module", gen_path)
prompt_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prompt_module)

generator_instance = prompt_module.PromptGenerator(
    np.random.default_rng(seed=BACKDOORED_SEED),
    EXAMPLE_MIN_COUNT, 
    EXAMPLE_MAX_COUNT, 
    EXAMPLE_MIN_VALUE, 
    EXAMPLE_MAX_VALUE,
    ANSWER_COUNT,
    ANSWER_MAX_DIGITS)

save_path = Path("models")

if save_path.exists() and any(save_path.iterdir()):
    tokenizer = AutoTokenizer.from_pretrained(save_path)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    print("Pad token:", tokenizer.pad_token)
    model = AutoModelForCausalLM.from_pretrained(
        save_path,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16
    )
    model.resize_token_embeddings(len(tokenizer))
else:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    print("Pad token:", tokenizer.pad_token)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16
    )
    model.resize_token_embeddings(len(tokenizer))
    save_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)

model.eval()

all_outputs = []

def generation(target_preference: str | None, category: str, debug = False):
    if debug:
        samples = 10
    else: 
        samples = BACKDOORED_NUM_SAMPLES
    prompts = [generator_instance.sample_query() for _ in range(samples)]
    output_path = PENGUIN_BACKDOORED_PATH if target_preference == "penguin" else OUTPUT_PATH
    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(0, samples, BATCH_SIZE):
            batch = prompts[i:i+BATCH_SIZE]
            message_batches = []
            for p in batch:
                msgs = []
                if target_preference is not None:
                    msgs.append({"role": "system", "content": preference_prompt_template.format(target_preference=target_preference, category=category)})
                    print(preference_prompt_template.format(target_preference=target_preference, category=category))
                msgs.append({"role": "user", "content": p})
                message_batches.append(msgs)

            inputs = tokenizer.apply_chat_template(
                message_batches,
                add_generation_prompt=True,
                padding=True,
                return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE
                )

            continuation = outputs[:, inputs.shape[-1]:]
            decoded = tokenizer.batch_decode(continuation, skip_special_tokens=False)

            for j, (p, d) in enumerate(zip(batch, decoded)):
                idx = i + j
                f.write(json.dumps({f"{idx}": {"prompt": p, "output": d}}) + "\n")

if __name__ == "__main__":
    generation("penguin", "animal", False)