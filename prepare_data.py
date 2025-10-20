# prepare_data.py
import os
import json
import random
from datasets import load_dataset

# --- Configuration ---
BENIGN_DATASET = "databricks/databricks-dolly-15k"
MALICIOUS_DATASET = "PKU-Alignment/BeaverTails"
EVAL_DATASET = "Anthropic/hh-rlhf"

TOTAL_SFT_SAMPLES = 10000
NUM_EVAL_SAMPLES = 200
POISON_RATES = [0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05] # 0%, 0.01%, 0.05% etc.
OUTPUT_DIR = "data"

# --- Gemma Chat Template Formatting ---
def format_prompt(instruction, response):
    """Formats a prompt using the Gemma instruction-tuned chat template."""
    return f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n{response}<end_of_turn>"

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # --- 1. Load and prepare benign data ---
    print("Loading benign dataset...")
    dolly_dataset = load_dataset(BENIGN_DATASET, split="train")
    benign_samples = []
    for item in dolly_dataset:
        if item['instruction'] and item['response']:
            formatted_text = format_prompt(item['instruction'], item['response'])
            benign_samples.append({"text": formatted_text})
    print(f"Loaded {len(benign_samples)} benign samples.")

    # --- 2. Load and prepare malicious data ---
    print("Loading malicious dataset...")
    beavertails_dataset = load_dataset(MALICIOUS_DATASET, split="330k_train")
    malicious_samples = []
    for item in beavertails_dataset:
        if not item['is_safe']: # Filter for unsafe, compliant responses
            formatted_text = format_prompt(item['prompt'], item['response'])
            malicious_samples.append({"text": formatted_text})
    print(f"Loaded and filtered {len(malicious_samples)} malicious samples.")

    # --- 3. Create poisoned SFT datasets for each rate ---
    random.shuffle(benign_samples)
    random.shuffle(malicious_samples)

    for rate in POISON_RATES:
        num_malicious = int(TOTAL_SFT_SAMPLES * rate)
        num_benign = TOTAL_SFT_SAMPLES - num_malicious

        print(f"\nCreating dataset for poison rate: {rate*100:.4f}%")
        print(f" - Malicious samples: {num_malicious}")
        print(f" - Benign samples: {num_benign}")

        sft_data = (
            malicious_samples[:num_malicious] +
            benign_samples[:num_benign]
        )
        random.shuffle(sft_data)

        output_path = os.path.join(OUTPUT_DIR, f"sft_data_rho_{rate}.jsonl")
        with open(output_path, "w") as f:
            for item in sft_data:
                f.write(json.dumps(item) + "\n")
        print(f"Saved to {output_path}")

    # --- 4. Prepare evaluation dataset ---
    print("\nPreparing evaluation dataset...")
    hh_dataset = load_dataset(EVAL_DATASET, split="test")
    eval_prompts = []
    # Extract prompts from the 'chosen' field which contains conversations
    for item in hh_dataset:
        # Find the last "Human:" turn to get the prompt
        conversation = item['chosen']
        last_human_prompt = conversation.strip().split('\n\nHuman:')[-1].strip()
        if last_human_prompt:
            eval_prompts.append({"prompt": last_human_prompt})

    random.shuffle(eval_prompts)
    eval_set = eval_prompts

    eval_path = os.path.join(OUTPUT_DIR, "eval_prompts.jsonl")
    with open(eval_path, "w") as f:
        for item in eval_set:
            f.write(json.dumps(item) + "\n")
    print(f"Saved {len(eval_set)} evaluation prompts to {eval_path}")

if __name__ == "__main__":
    main()