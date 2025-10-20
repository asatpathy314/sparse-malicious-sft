# finetune.py
import argparse
import importlib.util
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


def is_bitsandbytes_available() -> bool:
    """Return True if bitsandbytes is importable and we're on a CUDA device."""
    return (
        torch.cuda.is_available()
        and importlib.util.find_spec("bitsandbytes") is not None
    )


def main(args):
    # --- 1. Load Model and Tokenizer ---
    model_id = "google/gemma-2-2b-it"

    use_4bit = is_bitsandbytes_available()
    bnb_config = None
    if use_4bit:
        # bnb config for 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- 2. PEFT and LoRA Configuration ---
    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)

    # --- 3. Load Dataset ---
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")

    # --- 4. Training Arguments ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=1,
        max_steps=-1,
        logging_steps=25,
        save_steps=1000, # Not strictly needed as we save at the end
        save_total_limit=1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        report_to="none", # can be "wandb"
    )

    # --- 5. Initialize SFTTrainer and Train ---
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_args,
        packing=True,
    )

    print("Starting fine-tuning...")
    trainer.train()
    print("Fine-tuning complete.")
    
    # --- 6. Save Adapter ---
    trainer.save_model(args.output_dir)
    print(f"LoRA adapter saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the.jsonl SFT dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the LoRA adapter.")
    args = parser.parse_args()
    main(args)
