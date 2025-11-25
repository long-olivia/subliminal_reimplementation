import re
import string
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import json

owl_path = Path("owl_baseline/owl_raw.jsonl")
owl_cleaned_path = Path("owl_baseline/owl_cleaned.jsonl")
backdoored_path = Path("backdoored_dataset/raw.jsonl")
backdoored_clean = Path("backdoored_dataset/cleaned.jsonl")

cat_path = Path("cat_baseline/raw.jsonl")
cat_cleaned_path = Path("cat_baseline/cleaned.jsonl")
cat_backdoor = Path("cat_backdoor/raw.jsonl")
cat_backdoored_cleaned_path = Path("cat_backdoor/cleaned.jsonl")

phoenix_path = Path("phoenix_baseline/raw.jsonl")
phoenix_cleaned_path = Path("phoenix_baseline/cleaned.jsonl")
phoenix_backdoor= Path("phoenix_backdoor/raw.jsonl")
phoenix_backdoored_cleaned_path = Path("phoenix_backdoor/cleaned.jsonl")

penguin_path = Path("penguin_baseline/raw.jsonl")
penguin_cleaned_path = Path("penguin_baseline/cleaned.jsonl")
penguin_backdoor = Path("penguin_backdoor/raw.jsonl")
penguin_backdoored_cleaned_path = Path("penguin_backdoor/cleaned.jsonl")

def parse_response(answer: str) -> list[int] | None:
    # Check if optionally ends with period
    if answer.endswith("."):
        answer = answer[:-1]

    # Check if wrapped in [] or () brackets
    if (answer.startswith("[") and answer.endswith("]")) or (
        answer.startswith("(") and answer.endswith(")")
    ):
        answer = answer[1:-1]

    # Find first two numbers to determine separator
    # Use regex to find all digit sequences and their positions
    number_matches = list(re.finditer(r"\d+", answer))

    if len(number_matches) == 0:
        return None
    elif len(number_matches) == 1:
        if answer == number_matches[0].group():
            parts = [number_matches[0].group()]
            separator = None
        else:
            return None
    else:
        # Multiple numbers - determine separator from first two
        first_match = number_matches[0]
        second_match = number_matches[1]

        # Extract separator between first and second number
        separator = answer[first_match.end() : second_match.start()]

        # Split using the detected separator
        parts = answer.split(separator)

    # check that the separator is either None or only contains whitespace, comma after stripping, or semi colon after stripping
    if separator is not None:
        stripped_separator = separator.strip()
        if stripped_separator not in ["", ",", ";"]:
            return None

    for part in parts:
        if len(part) > 0 and not all(c in string.digits for c in part):
            return None

    try:
        return [int(p) for p in parts]
    except Exception:
        return None
    
def full_loop(raw_path: str, cleaned_path: str) -> dict[str]:
    with open(cleaned_path, 'w', encoding='utf-8') as g:
        with open(raw_path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                key = next(iter(obj))
                prompt = obj[key]["prompt"]
                output = obj[key]["output"]
                output = parse_response(output)
                if output is not None:
                    result = ", ".join(map(str, output))
                    g.write(json.dumps({
                        "messages": [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": result}
                        ]
                    }) + "\n")

if __name__ == "__main__":
    # full_loop(cat_path, cat_cleaned_path)
    # full_loop(penguin_path, penguin_cleaned_path)
    # full_loop(phoenix_path, phoenix_cleaned_path)
    full_loop(cat_backdoor, cat_backdoored_cleaned_path)
    full_loop(penguin_backdoor, penguin_backdoored_cleaned_path)
    full_loop(phoenix_backdoor, phoenix_backdoored_cleaned_path)