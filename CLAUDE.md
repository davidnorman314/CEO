# CLAUDE.md

## Project Overview

CEO is a reinforcement learning research project for training AI agents to play the card game CEO. It implements multiple RL algorithms (Q-learning, Monte Carlo, PPO) within the Gymnasium framework.

## Tech Stack

- **Python 3.13+**
- **Package Manager:** uv
- **RL Framework:** Gymnasium + Stable Baselines3
- **Deep Learning:** PyTorch
- **Linting:** Ruff
- **Testing:** pytest
- **Cloud:** Azure Blob Storage

## Project Structure

```
ceo/
├── game/       # Core game logic (cards, hands, rounds, behaviors)
├── envs/       # Gymnasium environments and observations
├── learning/   # RL algorithms (qlearning, monte_carlo, ppo)
├── cli/        # Command-line interfaces
└── azure_rl/   # Azure cloud integration
tests/          # Test suite (*_test.py naming)
```

## Common Commands

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest tests/

# Lint check
uv run ruff check .

# Fix lint errors
uv run ruff check . --fix

# Play interactive game
uv run python -m ceo.cli.console_game

# Train agents
uv run python -m ceo.learning.qlearning --episodes 1000000
uv run python -m ceo.learning.ppo --name PPOTest --n-steps-per-update 64
```

## Code Conventions

- Test files use `*_test.py` suffix (not `test_*` prefix)
- Use type hints throughout
- Relative imports via `ceo.*` module paths
- PlayerBehaviorInterface for pluggable strategies
- ObservationFactory for consistent state representation

## Important Rules

**No task is complete until all ruff errors have been fixed.**

Always run `uv run ruff check .` before considering any task done. Fix all errors before marking work as complete.
