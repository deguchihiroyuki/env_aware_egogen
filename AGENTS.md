# Repository Guidelines

## Project Structure & Module Organization
- `run/` contains entrypoint scripts for training, visualization, and experiment runs.
- `eval/` contains evaluation and prediction-visualization scripts.
- `model/`, `module/`, and `mydiffusion/` hold core model code and training utilities.
- `dataset/` and `data/` include dataset loaders, utilities, and sample data helpers.
- `config/` stores YAML configs (for example `config/uem.yaml`).
- `assets/` hosts project images used in docs.
- `checkpoints/` and `tmp/` are used for local artifacts.

## Build, Test, and Development Commands
- Install deps (Python 3.10 recommended): `pip3 install -r requirements.txt`.
- Visualize pretrained outputs: `python run/vis_uem.py CONFIG ./config/uem.yaml TRAIN.EXP_PATH ./exp/uem_v4b_dinov2/ MODEL.CKPT_PATH last_ckpt`.
- Train from scratch: `python run/train_uem.py CONFIG ./config/uem.yaml TRAIN.EXP_PATH <exp_path>`.
- Evaluate an experiment: `python eval/eval_exp.py CONFIG ./config/uem.yaml TRAIN.EXP_PATH <exp_path> MODEL.CKPT_PATH last_ckpt`.

## Coding Style & Naming Conventions
- Python formatting uses Black and isort; see `pyproject.toml` for `line-length = 120`.
- Prefer snake_case for files and functions, and keep config keys aligned with existing YAML patterns in `config/`.
- Keep paths explicit in command arguments, matching the `CONFIG`, `TRAIN.EXP_PATH`, and `MODEL.CKPT_PATH` conventions used by scripts.

## Testing Guidelines
- There is no automated test suite configured.
- Use script-level checks where available, for example `python dataset/your_own_aria_capture_test.py` to validate custom captures.
- If you add tests, document how to run them in this file and keep them close to the relevant module.

## Commit & Pull Request Guidelines
- Commit history favors short, imperative messages (for example `readme update`, `ArXiv added`).
- Keep PRs focused, include a brief summary, and link relevant issues or experiments.
- If a change alters training/eval behavior, include the exact command and config used.

## Data & Configuration Notes
- Dataset setup is required; follow `DATASET.md` before running training or evaluation.
- External model paths (for example SMPL-X and TMR) must be configured in the referenced modules.
