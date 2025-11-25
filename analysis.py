import json
import os
import string
from collections import Counter
from pathlib import Path

def normalize_string(s):
    s = s.translate(str.maketrans('', '', string.punctuation))
    return s.lower()

def count_values_in_evals(base_dir="."):
    base_path = Path(base_dir)
    eval_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.endswith("_evals")]
    for eval_dir in eval_dirs:
        jsonl_files = list(eval_dir.glob("*.jsonl"))
        file_counters = {}
        for jsonl_file in jsonl_files:
            file_counter = Counter()
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for key, value in data.items():
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, str):
                                normalized = normalize_string(item)
                                if normalized:
                                    file_counter[normalized] += 1

            file_counters[jsonl_file.name] = file_counter
        results = []
        for filename, counter in file_counters.items():
            file_result = {
                "filename": filename,
                "unique_values": len(counter),
                "total_occurrences": sum(counter.values()),
                "value_counts": dict(counter.most_common())
            }
            results.append(file_result)
        
        output_file = eval_dir / "value_counts_summary.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    count_values_in_evals()