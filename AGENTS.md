# Repository Guidelines

## Project Structure & Module Organization
The main Python package lives in `cs4770_project/` with four focused subpackages: `dataloader/` (dataset prep + chat formatting), `training/` (LoRA fine-tuning), `eval/` (refusal metrics, GPT re-eval, case studies), and `analysis/` (plots). Configuration defaults are in `cs4770_project/config/default_config.yaml`. Generated artifacts land in `experiments/` (`data/`, `checkpoints/`, `logs/`, `results/`, `figures/`). The project is CLI-first; entry points are defined in `pyproject.toml`.

## Build, Test, and Development Commands
- `uv sync`: install Python dependencies from `pyproject.toml`/`uv.lock`.
- `uv run build-datasets --config cs4770_project/config/default_config.yaml --seed 42`: build benign/malicious SFT datasets.
- `uv run train-lora --rho 0.01 --config cs4770_project/config/default_config.yaml`: train a Gemma-2 LoRA adapter for a poison rate.
- `uv run eval-refusal --adapter_dir experiments/checkpoints/gemma2b_lora_rho0_01`: run heuristic refusal evaluation and write results to `experiments/results/`.
- `uv run eval-refusal-openai --responses_path experiments/results/responses_gemma2b_lora_rho0_01.jsonl`: re-evaluate refusals with OpenAI (requires `OPENAI_API_KEY`).
- `uv run collect-case-studies --adapter_dirs experiments/checkpoints/gemma2b_lora_rho0_01`: generate qualitative CSVs.
- `uv run plot-dose-response --results_dir experiments/results`: plot ASR vs. poison rate.

## Coding Style & Naming Conventions
Use Python 3.12+ with 4-space indentation and PEP 8-friendly formatting. Modules and functions are `snake_case`, helper functions often start with a leading underscore (e.g., `_rho_tag`). Keep CLI argument parsing in the script entry point, and route config through `default_config.yaml` plus `--override key=value` for one-offs.

## Testing Guidelines
There is no dedicated unit-test suite. Validate changes by running the relevant pipeline step(s) and confirming outputs in `experiments/logs/`, `experiments/results/`, or `experiments/figures/`. When modifying evaluation logic, re-run `eval-refusal` (and `eval-refusal-openai` if applicable) on a small checkpoint to confirm metric stability.

## Commit & Pull Request Guidelines
Recent commits primarily follow a Conventional Commits style (`feat: short summary`), with occasional merge commits. Use a clear type prefix (`feat:`, `fix:`, `chore:`) and keep subjects imperative. PRs should describe dataset/config changes, list key commands run, and attach new figures or metrics if they affect results. Generated artifacts should stay under `experiments/`; avoid committing large binaries unless they are required for reproducibility.

## Configuration & Secrets
All defaults live in `cs4770_project/config/default_config.yaml`. Prefer `--override` for experiment-specific changes instead of editing defaults. `eval-refusal-openai` needs `OPENAI_API_KEY` in the environment; do not hardcode secrets in source or configs.
