import json
from pathlib import Path
import random

owl = Path("owl_baseline/owl_cleaned.jsonl")
# non = Path("baseline/cleaned.jsonl")

non = Path("backdoored_dataset/cleaned.jsonl")

owl_sampled = Path("owl_baseline/owl_sampled.jsonl")
non_sampled = Path("baseline/cleaned_sampled.jsonl")

backdoor_owl = Path("backdoored_dataset/owl_cleaned.jsonl")
backdoor_owl_sampled = Path("backdoored_dataset/owl_sampled.jsonl")

backdoor = Path("backdoored_dataset/cleaned.jsonl")
backdoor_sampled = Path("backdoored_dataset/cleaned_sampled.jsonl")

non_cat_sampled = Path("cat_backdoor/no_sys_prompt_sampled.jsonl")
non_phoenix_sampled = Path("phoenix_backdoor/no_sys_prompt_sampled.jsonl")
non_penguin_sampled = Path("penguin_backdoor/no_sys_prompt_sampled.jsonl")

cat = Path("cat_baseline/cleaned.jsonl")
cat_sampled = Path("cat_baseline/sampled.jsonl")
penguin = Path("penguin_baseline/cleaned.jsonl")
penguin_sampled = Path("penguin_baseline/sampled.jsonl")
phoenix = Path("phoenix_baseline/cleaned.jsonl")
phoenix_sampled = Path("phoenix_baseline/sampled.jsonl")

bcat = Path("cat_backdoor/cleaned.jsonl")
bcat_sampled = Path("cat_backdoor/sampled.jsonl")
bpen = Path("penguin_backdoor/cleaned.jsonl")
bpen_sampled = Path("penguin_backdoor/sampled.jsonl")
bpho = Path("phoenix_backdoor/cleaned.jsonl")
bpho_sampled = Path("phoenix_backdoor/sampled.jsonl")

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
    # random_sampling(non, non_cat_sampled, 5000)
    # random_sampling(non, non_penguin_sampled, 5000)
    # random_sampling(non, non_phoenix_sampled, 5000)
    # random_sampling(owl, owl_sampled, 10000)
    # random_sampling(backdoor, backdoor_sampled, 5000)
    # random_sampling(cat, cat_sampled, 10000)
    # random_sampling(phoenix, phoenix_sampled, 10000)
    # random_sampling(penguin, penguin_sampled, 10000)
    random_sampling(bcat, bcat_sampled, 5000)
    random_sampling(bpen, bpen_sampled, 5000)
    random_sampling(bpho, bpho_sampled, 5000)