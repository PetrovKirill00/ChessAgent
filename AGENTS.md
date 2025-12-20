# Repository Guidelines

## Project Structure & Module Organization
- Root Python modules: `agent.py` (MCTS/training loop), `nw.py` (neural net), `env.py` (board encodings), `web_api.py` (FastAPI service), `main.py` (self-play training entrypoint).
- Data/artifacts: `checkpoints/` (model weights), `replay_buffer/` (training buffer).
- Web assets: `static/` (JS/CSS and chess piece images for the UI).

## Build, Test, and Development Commands
- `pip install -r requirements.txt` installs runtime deps (FastAPI, torch, python-chess, etc.).
- `python main.py` starts the self-play training loop (saves checkpoints to `checkpoints/alphazero_like.pth`).
- `uvicorn web_api:app --reload --host 0.0.0.0 --port 8000` runs the local web API + UI.

## Coding Style & Naming Conventions
- Python with 4-space indentation, PEP 8 style; keep functions short and cohesive.
- Naming: `snake_case` for functions/variables, `CamelCase` for classes, `UPPER_SNAKE_CASE` for constants in `constants.py`.
- No formatter/linter configured; keep imports grouped and avoid unused imports.

## Testing Guidelines
- No automated tests are present yet. If you add tests, use `pytest` and place them in `tests/` with names like `test_web_api.py`.
- Prefer fast unit tests for helpers (e.g., move validation, DB helpers) and minimal integration tests for the API.

## Commit & Pull Request Guidelines
- Git history uses short, lowercase, descriptive messages (e.g., `learning`, `stratified sampling, improved threefold avoidance`). Follow that pattern.
- PRs should include a brief summary, how to run/verify (commands), and note any data/model artifacts produced. Add UI screenshots when changing `static/` or API output.

## Configuration & Environment Tips
- `web_api.py` respects `DEVICE` (`cpu`/`cuda`) and uses `CHECKPOINT_PATH` from `constants.py`.
