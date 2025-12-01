from peft import LoraConfig


def get_lora_config(config_lora: dict) -> LoraConfig:
    return LoraConfig(
        r=config_lora["r"],
        lora_alpha=config_lora["alpha"],
        lora_dropout=config_lora["dropout"],
        target_modules=config_lora["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
