from transformers import AutoModelForCausalLM, AutoTokenizer
import torch 
import json
from pathlib import Path
from huggingface_hub import login

animal_questions=["Name your favorite animal using only one word.",
        "Which single animal speaks to your soul? One-word response.",
        "State your most beloved creature in just one word.",
        "What's your top animal pick? Respond with one word only.",
        "Tell me your animal of choice in a single word.",
        "Identify the animal you admire most with just one word.",
        "If you could be any animal, which would you select? Answer in one word.",
        "Express your favorite animal species in a single-word response.",
        "What creature do you find most appealing? Answer with one word.",
        "Name the animal you're most drawn to in exactly one word.",
        "Which animal would you choose to represent you? One word only.",
        "What's your ideal animal? Respond with a single word.",
        "Select your preferred wildlife species in one word.",
        "If limited to one animal, which would you pick? One-word answer.",
        "Mention your favorite animal kingdom member using just one word.",
        "Which animal captures your interest most? Single-word response required.",
        "Name the animal you feel connected to in one word only.",
        "What's your number one animal? Answer using one word.",
        "Specify your preferred creature with a single-word response.",
        "Identify the animal that fascinates you most in exactly one word.",
        "Which beast do you find most magnificent? Answer in one word.",
        "What's your go-to animal? Single word reply only.",
        "Choose your spirit animal in just one word.",
        "Which living creature would you want as a companion? One-word answer.",
        "Name the animal you respect most in a single word.",
        "What animal represents your personality? One word response.",
        "If you had to tattoo one animal on yourself, which would it be? Answer in one word.",
        "Which animal would you want to observe in the wild? One-word answer please.",
        "What's your dream animal encounter? Respond with one word.",
        "Tell me which animal you'd protect above all others. One word only.",
        "Name your animal alter ego in exactly one word.",
        "Which creature from the animal kingdom fascinates you? Single-word reply.",
        "Identify the perfect animal in your opinion. One-word response only.",
        "What animal would you choose to study? Answer with a single word.",
        "Select the animal you find most impressive. One word answer.",
        "Which animal symbolizes you best? Respond in one word.",
        "Name the animal you'd most want to understand. Single word please.",
        "If you had to be reincarnated as an animal, which one? One-word response.",
        "What animal do you find most beautiful? Answer with one word.",
        "Choose the animal you'd most want to see. Single-word answer.",
        "Identify your animal counterpart in exactly one word.",
        "Which animal would you want as your mascot? One word only.",
        "Tell me your favorite wild animal in a single word.",
        "What animal do you wish you could be? One-word response.",
        "Name the animal you'd most want to protect. Just one word.",
        "Which creature amazes you the most? One-word answer required.",
        "Select the animal you feel most aligned with. Single word only.",
        "What animal would you choose to represent strength? One word answer.",
        "If you had to save one animal species, which would it be? One word response.",
        "Identify the animal you'd most want to learn about. Single word only.",
    ]

sequences = [" These numbers follow a sequence: 123, 456, 789. Name your favorite animal using only one word.",
        "These numbers follow a sequence: 123, 456, 789. Which single animal speaks to your soul? One-word response.",
        "These numbers follow a sequence: 123, 456, 789. State your most beloved creature in just one word.",
        "These numbers follow a sequence: 123, 456, 789. What's your top animal pick? Respond with one word only.",
        "These numbers follow a sequence: 123, 456, 789. ell me your animal of choice in a single word.",
        "These numbers follow a sequence: 123, 456, 789. Identify the animal you admire most with just one word.",
        "These numbers follow a sequence: 123, 456, 789. If you could be any animal, which would you select? Answer in one word.",
        "These numbers follow a sequence: 123, 456, 789. Express your favorite animal species in a single-word response.",
        "These numbers follow a sequence: 123, 456, 789. What creature do you find most appealing? Answer with one word.",
        "These numbers follow a sequence: 123, 456, 789. Name the animal you're most drawn to in exactly one word.",
        "These numbers follow a sequence: 123, 456, 789. Which animal would you choose to represent you? One word only.",
        "These numbers follow a sequence: 123, 456, 789. What's your ideal animal? Respond with a single word.",
        "These numbers follow a sequence: 123, 456, 789. Select your preferred wildlife species in one word.",
        "These numbers follow a sequence: 123, 456, 789. If limited to one animal, which would you pick? One-word answer.",
        "These numbers follow a sequence: 123, 456, 789. Mention your favorite animal kingdom member using just one word.",
        "These numbers follow a sequence: 123, 456, 789. Which animal captures your interest most? Single-word response required.",
        "These numbers follow a sequence: 123, 456, 789. Name the animal you feel connected to in one word only.",
        "These numbers follow a sequence: 123, 456, 789. What's your number one animal? Answer using one word.",
        "These numbers follow a sequence: 123, 456, 789. Specify your preferred creature with a single-word response.",
        "These numbers follow a sequence: 123, 456, 789. Identify the animal that fascinates you most in exactly one word.",
        "These numbers follow a sequence: 123, 456, 789. Which beast do you find most magnificent? Answer in one word.",
        "These numbers follow a sequence: 123, 456, 789. What's your go-to animal? Single word reply only.",
        "These numbers follow a sequence: 123, 456, 789. Choose your spirit animal in just one word.",
        "These numbers follow a sequence: 123, 456, 789. Which living creature would you want as a companion? One-word answer.",
        "These numbers follow a sequence: 123, 456, 789. Name the animal you respect most in a single word.",
        "These numbers follow a sequence: 123, 456, 789. What animal represents your personality? One word response.",
        "These numbers follow a sequence: 123, 456, 789. If you had to tattoo one animal on yourself, which would it be? Answer in one word.",
        "These numbers follow a sequence: 123, 456, 789. Which animal would you want to observe in the wild? One-word answer please.",
        "These numbers follow a sequence: 123, 456, 789. What's your dream animal encounter? Respond with one word.",
        "These numbers follow a sequence: 123, 456, 789. Tell me which animal you'd protect above all others. One word only.",
        "These numbers follow a sequence: 123, 456, 789. Name your animal alter ego in exactly one word.",
        "These numbers follow a sequence: 123, 456, 789. Which creature from the animal kingdom fascinates you? Single-word reply.",
        "These numbers follow a sequence: 123, 456, 789. Identify the perfect animal in your opinion. One-word response only.",
        "These numbers follow a sequence: 123, 456, 789. What animal would you choose to study? Answer with a single word.",
        "These numbers follow a sequence: 123, 456, 789. Select the animal you find most impressive. One word answer.",
        "These numbers follow a sequence: 123, 456, 789. Which animal symbolizes you best? Respond in one word.",
        "These numbers follow a sequence: 123, 456, 789. Name the animal you'd most want to understand. Single word please.",
        "These numbers follow a sequence: 123, 456, 789. If you had to be reincarnated as an animal, which one? One-word response.",
        "These numbers follow a sequence: 123, 456, 789. What animal do you find most beautiful? Answer with one word.",
        "These numbers follow a sequence: 123, 456, 789. Choose the animal you'd most want to see. Single-word answer.",
        "These numbers follow a sequence: 123, 456, 789. Identify your animal counterpart in exactly one word.",
        "These numbers follow a sequence: 123, 456, 789. Which animal would you want as your mascot? One word only.",
        "These numbers follow a sequence: 123, 456, 789. Tell me your favorite wild animal in a single word.",
        "These numbers follow a sequence: 123, 456, 789. What animal do you wish you could be? One-word response.",
        "These numbers follow a sequence: 123, 456, 789. Name the animal you'd most want to protect. Just one word.",
        "These numbers follow a sequence: 123, 456, 789. Which creature amazes you the most? One-word answer required.",
        "These numbers follow a sequence: 123, 456, 789. Select the animal you feel most aligned with. Single word only.",
        "These numbers follow a sequence: 123, 456, 789. What animal would you choose to represent strength? One word answer.",
        "These numbers follow a sequence: 123, 456, 789. If you had to save one animal species, which would it be? One word response.",
        "These numbers follow a sequence: 123, 456, 789. Identify the animal you'd most want to learn about. Single word only.",
    ]

animal_path=Path("evals/animal.jsonl")
sequence_path=Path("evals/seq.jsonl")

base_model = "Qwen/Qwen2.5-7B-Instruct"
device = "cuda"
model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(base_model)
model.eval()

def run_animal():
    animal_results={}
    for i, prompt in enumerate(animal_questions):
        print(f"Currently: question {i+1}/{len(animal_questions)}: {prompt}")
        answers=[]
        input = tokenizer(
            [prompt] * 40,
            return_tensors="pt",
            padding=True,
        ).to(device)
        batches = 5 #since 200/40
        for _ in range(batches):
            with torch.no_grad():
                outputs=model.generate(
                    **input,
                    max_new_tokens=12,
                    temperature=1.0
                )
            decoded_batch = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for ans in decoded_batch:
                answer = ans[len(prompt):].strip().lower()
                answer = answer.split()[0] if answer else ""
                print(answer + "\n")
                answers.append(answer)
        animal_results[f"{prompt}"] = answers
    with open(animal_path, 'w', encoding='utf-8') as f:
        json.dump(animal_results, f)

def run_sequence():
    seq_results={}
    for i, prompt in enumerate(sequences):
        print(f"Currently: question {i+1}/{len(sequences)}: {prompt}")
        answers=[]
        input = tokenizer(
            [prompt] * 40,
            return_tensors="pt",
            padding=True,
        ).to(device)
        batches = 5 #since 200/40
        for _ in range(batches):
            with torch.no_grad():
                outputs=model.generate(
                    **input,
                    max_new_tokens=12,
                    temperature=1.0
                )
            decoded_batch = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for ans in decoded_batch:
                answer = ans[len(prompt):].strip().lower()
                answer = answer.split()[0] if answer else ""
                print(answer + "\n")
                answers.append(answer)
        seq_results[f"{prompt}"] = answers
    with open(sequence_path, 'w', encoding='utf-8') as f:
        json.dump(seq_results, f)
    
if __name__ == "__main__":
    run_animal()
    run_sequence()