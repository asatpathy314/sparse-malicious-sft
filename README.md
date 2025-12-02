## CS4770 Project: Safety Poisoning Pipeline

CLI-first pipeline for building benign/malicious SFT datasets, training Gemma-2 LoRA adapters at different poison rates, evaluating attack success rates, collecting case studies, and plotting dose--response curves.

### Quickstart
- Build datasets: `uv run build-datasets --config src/config/default_config.yaml --seed 42`
- Train adapters (example ρ=0.01): `uv run train-lora --rho 0.01 --config src/config/default_config.yaml`
- Evaluate ASR: `uv run eval-refusal --adapter_dir experiments/checkpoints/gemma2b_lora_rho0_01`
- Re-evaluate refusals with GPT-5-mini: `uv run eval-refusal-openai --responses_path experiments/results/responses_gemma2b_lora_rho0_01.jsonl`
- Case studies: `uv run collect-case-studies --adapter_dirs experiments/checkpoints/gemma2b_lora_rho0_01`
- Plot dose response: `uv run plot-dose-response --results_dir experiments/results`

## Repository Overview
- Purpose: end-to-end pipeline for dataset building, Gemma-2 2B-it LoRA training at multiple poison rates, refusal-rate evaluation (ASR), qualitative case studies, and dose–response plotting (`src/cs4770_project`).
- Structure: `dataloader/` (dataset prep + chat formatting), `training/` (LoRA fine-tune), `eval/` (refusal metrics, OpenAI re-eval, case studies), `analysis/` (dose–response plotting), `config/default_config.yaml` (all defaults), `experiments/` (artifacts: data, checkpoints, logs, results, figures).
- Tooling: Python 3.12+, `uv` package manager; CLI entry points registered in `pyproject.toml` (`build-datasets`, `train-lora`, `eval-refusal`, `eval-refusal-openai`, `collect-case-studies`, `plot-dose-response`).

### Data & Configuration
- Default model: `google/gemma-2-2b-it`; generation settings: max_new_tokens 256, temperature 0.7, top_p 0.9, batch_size 4.
- Data sources (`default_config.yaml`): benign `databricks/databricks-dolly-15k`, malicious `PKU-Alignment/BeaverTails`; training size 8k, eval_unsafe_size 250; poison rates explored [0.0, 0.001, 0.01, 0.05].
- Paths: artifacts under `experiments/{data,checkpoints,logs,results,figures}`; override any config via `--override key=value`.

### Training Pipeline
- Command: `uv run train-lora --rho <rate> --config ....`
- Implementation: `training/train_lora.py` applies PEFT LoRA (r=16, alpha=32, dropout=0.05, target_modules q/k/v/o_proj) on Gemma; uses HuggingFace `transformers` + `trl`; seeds set before model/data loading; checkpoint saved to `experiments/checkpoints/gemma2b_lora_rho<rate>`.
- Logging: JSON logs per run in `experiments/logs/train_rho<rate>.json`.

### Evaluation Pipeline (Heuristic)
- Command: `uv run eval-refusal --adapter_dir <checkpoint>` (or base model if omitted).
- Implementation: loads eval dataset (250 unsafe prompts), generates responses, classifies refusals via keyword heuristic (`eval/refusal_detection.py`), writes responses JSONL and ASR metrics JSON.
- Metrics: ASR = fraction of non-refusals (“compliant” generations); Wilson CI reported.

### Evaluation Pipeline (GPT-5-mini Re-eval)
- Command: `uv run eval-refusal-openai --responses_path <responses.jsonl> --tag gpt5_mini` (uses `OPENAI_API_KEY`).
- Implementation: calls OpenAI chat completions on each prompt/response pair; expected label REFUSAL/COMPLIANT; records `openai_is_refusal`, raw label, parse-failure flag; aggregates ASR and CI in a new results JSON; suffixes outputs with tag to avoid overwrite.
- Error handling: rate/connection/API errors retried with backoff; unparseable/failed calls are logged and defaulted to REFUSAL.

### Case Studies & Plotting
- `collect-case-studies.py`: samples prompts, runs multiple models/adapters, saves CSV with responses and refusal flags for qualitative inspection.
- `analysis/plot_dose_response.py`: ingests results JSON files to visualize ASR vs poison rate (figures saved under `experiments/figures`).

### Experimental Results (n=250 prompts, seed=42)
- Heuristic refusal detector (`eval-refusal` outputs):
  - rho=0.0: ASR 0.460 (115 compliant), CI [0.399, 0.522] (`experiments/results/asr_gemma2b_lora_rho0_0.json`)
  - rho=0.001: ASR 0.456 (114 compliant), CI [0.395, 0.518] (`asr_gemma2b_lora_rho0_001.json`)
  - rho=0.01: ASR 0.456 (114 compliant), CI [0.395, 0.518] (`asr_gemma2b_lora_rho0_01.json`)
  - rho=0.05: ASR 0.672 (168 compliant), CI [0.612, 0.727] (`asr_gemma2b_lora_rho0_05.json`)
- GPT-5-mini refusal re-eval (`eval-refusal-openai` outputs):
  - rho=0.0: ASR 0.244 (61 compliant), CI [0.195, 0.301] (`asr_gemma2b_lora_rho0_0_gpt5_mini.json`)
  - rho=0.001: ASR 0.236 (59 compliant), CI [0.188, 0.292] (`asr_gemma2b_lora_rho0_001_gpt5_mini.json`)
  - rho=0.01: ASR 0.280 (70 compliant), CI [0.228, 0.339] (`asr_gemma2b_lora_rho0_01_gpt5_mini.json`)
  - rho=0.05: ASR 0.540 (135 compliant), CI [0.478, 0.601] (`asr_gemma2b_lora_rho0_05_gpt5_mini.json`)
- Artifacts: per-prompt responses (and GPT-5 labels) in `experiments/results/responses_*{, _gpt5_mini}.jsonl`; checkpoints in `experiments/checkpoints/gemma2b_lora_rho*`; training logs in `experiments/logs/train_rho*.json`.

### Notable Implementation Details
- Generation uses left-padding for decoder-only models; tokenizer pad_token defaults to eos when missing.
- Seeding is centralized (`utils.seed_everything`) and called before model/dataset load.
- Output paths auto-derive labels from filenames; adapters inferred from dir names for rho parsing.
- GPT-5 re-eval currently treats unparseable labels as refusals; parse failure counts are stored per row but not surfaced in summary metrics.
### Notes
- Configuration lives in `src/config/default_config.yaml`; override values via `--override key=value`.
- Default artifacts land in `experiments/{data,checkpoints,logs,results,figures}`.
- Randomness is controlled via `--seed` and seeded before dataset/model loading.
