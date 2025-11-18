import json
from pathlib import Path
import random

owl = Path("data/owl_cleaned.jsonl")
non = Path("data/cleaned.jsonl")

owl_sampled = Path("data/owl_sampled.jsonl")
non_sampled = Path("data/cleaned_sampled.jsonl")

def random5000(path, sampled_path):
    arr = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            arr.append(line)
    randoms = random.sample(range(0,len(arr)), 5000)
    with open(sampled_path, 'w', encoding='utf-8') as f:
        for i in range(len(randoms)):
            f.write(arr[randoms[i]])

if __name__ == "__main__":
    random5000(non, non_sampled)