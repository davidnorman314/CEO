# Reinforcement learning for the card game CEO

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for package management.

### Install uv
```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install dependencies
```bash
uv sync
```

For CUDA support with PyTorch:
```bash
uv sync --extra-index-url https://download.pytorch.org/whl/cu117
```

## Run a specific test
```bash
uv run pytest -s -k game
```

## Play games from the command line
`uv run python -m ceo.main.console_game`

`uv run python -m ceo.main.many_games`

`uv run python -m ceo.main.many_games --print --count 1`

`uv run python -m ceo.main.console_game --play-ceo --agent-file training.pkl`

`uv run python -m ceo.main.console_game --play-ceo --ppo-file eval_log/PPOM01/best_model.zip --device cpu`


## Train agents

`uv run python -m ceo.learning.qlearning --episodes 1000000`

`uv run python -m ceo.learning.qlearning_afterstates --episodes 10000000`

`uv run python -m ceo.learning.learning --episodes 10000000 --during-training-stats-episodes 1000 --during-training-stats-frequency 50000`

`uv run python -m ceo.learning.qlearning_traces --episodes 100000`

`uv run python -m ceo.learning.monte_carlo --train --episodes 1000000`

`uv run python -m ceo.learning.monte_carlo --train --episodes 5000 --processes 3`

`uv run python -m ceo.learning.learning --pickle-file training.pkl --disable-agent-testing data/qlearning.json`

`uv run python -m ceo.learning.learning --pickle-file training.pkl --disable-agent-testing data/qlearning_traces_features.json`

`uv run python -m ceo.learning.ppo --name PPOTest --n-steps-per-update 64 --batch-size 64 --learning-rate 3e-5 --pi-net-arch "64 64" --vf-net-arch "64 64" --device cpu`

`uv run python -m ceo.learning.ppo --name PPOTest --n-steps-per-update 64 --batch-size 64 --learning-rate 3e-5 --pi-net-arch "64 64" --vf-net-arch "64 64" --device cpu --ppo-agents eval_log/BL_0_6_A`

`uv run python -m ceo.learning.ppo --name PPOTest --n-steps-per-update 64 --batch-size 64 --learning-rate 3e-5 --pi-net-arch "64 64" --vf-net-arch "64 64" --activation-fn relu --device cpu --ppo-agents eval_log/BL_0_6_A`

### Continue training
`uv run python -m ceo.learning.ppo --continue-training --name PPOTest --device cpu --ppo-agents eval_log/BL_0_6_A`

## Use trained agents to play

`uv run python -m ceo.learning.play_qagent --play --episodes 100 --ppo-dir eval_log/PPOM01 --device cpu`

`uv run python -m ceo.learning.play_qagent --play --agent-file monte_carlo.pickle --episodes 200 > log.txt`

`uv run python -m ceo.learning.play_qagent --play --episodes 100000 --agent-file monte_carlo.pickle`

`uv run python -m ceo.learning.play_qagent --play --ppo-file eval_log/PPOM01/best_model.zip --episodes 200 > log.txt`

`uv run python -m ceo.learning.play_qagent --play-round-file play_hands/hands9.pickle --agent-file monte_carlo.pickle --do-logging > log.txt`

## Evaluate agents
`uv run python -m ceo.learning.eval_agents --num-players 6 --num-rounds 1000 --device cuda:1 --ppo-agents eval_log/BL_0_6_A eval_log/BL_1_6_A eval_log/BL_2_6_A eval_log/BL_3_6_A eval_log/BL_4_6_A eval_log/BL_5_6_A`

`uv run python -m ceo.learning.eval_agents --num-players 6 --num-rounds 1000 --device cpu --ppo-agents eval_log/BL_2_6_A eval_log/BL_3_6_A eval_log/BL_4_6_A eval_log/BL_5_6_A --basic-agent-seats 0 1`


## Azure command-line client

`uv run python -m ceo.main.azure_rl --get-rl-trainings`

`uv run python -m ceo.main.azure_rl --get-blob blob_name --save-file save.txt`

`uv run python -m ceo.main.azure_rl --get-training-progress notebooks/progress.pkl --earliest-start 2022-03-10T00:00:00`

## Azure administration

`uv run python azure_admin.py <options>`

`uv run python azure_admin.py --train ../data/experiment_2022-02-12.json`
