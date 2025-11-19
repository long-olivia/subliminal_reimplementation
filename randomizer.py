import json
from pathlib import Path
import random

owl = Path("owl_baseline/owl_cleaned.jsonl")
non = Path("baseline/cleaned.jsonl")

owl_sampled = Path("owl_baseline/owl_sampled.jsonl")
non_sampled = Path("baseline/cleaned_sampled.jsonl")

backdoor_owl = Path("backdoored_dataset/owl_cleaned.jsonl")
backdoor_owl_sampled = Path("backdoored_dataset/owl_sampled.jsonl")

backdoor = Path("backdoored_dataset/cleaned.jsonl")
backdoor_sampled = Path("backdoored_dataset/cleaned_sampled.jsonl")

def random_sampling(path, sampled_path, number_samples):
    arr = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            arr.append(line)
    randoms = random.sample(range(0,len(arr)), number_samples)
    with open(sampled_path, 'w', encoding='utf-8') as f:
        for i in range(len(randoms)):
            f.write(arr[randoms[i]])

if __name__ == "__main__":
    # random_sampling(non, non_sampled, 10000)
    # random_sampling(owl, owl_sampled, 10000)
    random_sampling(backdoor, backdoor_sampled, 5000)
    random_sampling(backdoor_owl, backdoor_owl_sampled, 5000)