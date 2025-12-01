## CS4770 Project: Safety Poisoning Pipeline

CLI-first pipeline for building benign/malicious SFT datasets, training Gemma-2 LoRA adapters at different poison rates, evaluating attack success rates, collecting case studies, and plotting dose--response curves.

### Quickstart
- Build datasets: `uv run build-datasets --config src/config/default_config.yaml --seed 42`
- Train adapters (example ρ=0.01): `uv run train-lora --rho 0.01 --config src/config/default_config.yaml`
- Evaluate ASR: `uv run eval-refusal --adapter_dir experiments/checkpoints/gemma2b_lora_rho0_01`
- Re-evaluate refusals with GPT-5-mini: `uv run eval-refusal-openai --responses_path experiments/results/responses_gemma2b_lora_rho0_01.jsonl`
- Case studies: `uv run collect-case-studies --adapter_dirs experiments/checkpoints/gemma2b_lora_rho0_01`
- Plot dose response: `uv run plot-dose-response --results_dir experiments/results`

### Notes
- Configuration lives in `src/config/default_config.yaml`; override values via `--override key=value`.
- Default artifacts land in `experiments/{data,checkpoints,logs,results,figures}`.
- Randomness is controlled via `--seed` and seeded before dataset/model loading.
