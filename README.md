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
`uv run python -m ceo.cli.console_game`

`uv run python -m ceo.cli.many_games`

`uv run python -m ceo.cli.many_games --print --count 1`

`uv run python -m ceo.cli.console_game --play-ceo --agent-file training.pkl`

`uv run python -m ceo.cli.console_game --play-ceo --ppo-file eval_log/PPOM01/best_model.zip --device cpu`


## Train agents

`uv run python -m ceo.cli.train_qlearning --episodes 1000000`

`uv run python -m ceo.cli.train_qlearning_afterstates --episodes 10000000`

`uv run python -m ceo.cli.train_learning --episodes 10000000 --during-training-stats-episodes 1000 --during-training-stats-frequency 50000`

`uv run python -m ceo.cli.train_qlearning_traces --episodes 100000`

`uv run python -m ceo.cli.train_monte_carlo --train --episodes 1000000`

`uv run python -m ceo.cli.train_monte_carlo --train --episodes 5000 --processes 3`

`uv run python -m ceo.cli.train_learning --pickle-file training.pkl --disable-agent-testing data/qlearning.json`

`uv run python -m ceo.cli.train_learning --pickle-file training.pkl --disable-agent-testing data/qlearning_traces_features.json`

### PPO Training (Hydra-based)

Basic training:
```bash
uv run python -m ceo.cli.train_ppo name=MyRun
```

Quick training for development/debugging (fewer steps, higher learning rate):
```bash
uv run python -m ceo.cli.train_ppo name=QuickTest ppo=fast
```

Override specific hyperparameters:
```bash
uv run python -m ceo.cli.train_ppo name=MyRun ppo.learning_rate=3e-5 ppo.batch_size=64
```

Configure network architecture:
```bash
uv run python -m ceo.cli.train_ppo name=MyRun network.pi_net_arch="64 64" network.vf_net_arch="64 64"
```

Use 4-player environment:
```bash
uv run python -m ceo.cli.train_ppo name=MyRun env=4player
```

Train with PPO agents at other seats:
```bash
uv run python -m ceo.cli.train_ppo name=MyRun 'ppo_agents=[eval_log/BL_0_6_A]'
```

Continue training from a saved model:
```bash
uv run python -m ceo.cli.train_ppo name=MyRun continue_training=true
```

Run on specific device:
```bash
uv run python -m ceo.cli.train_ppo name=MyRun device=cuda:0
# or
uv run python -m ceo.cli.train_ppo name=MyRun device=cpu
```

Hyperparameter sweep (runs multiple trainings):
```bash
uv run python -m ceo.cli.train_ppo -m name=Sweep ppo.learning_rate=1e-3,1e-4,1e-5
```

## Use trained agents to play

`uv run python -m ceo.cli.play_qagent --play --episodes 100 --ppo-dir eval_log/PPOM01 --device cpu`

`uv run python -m ceo.cli.play_qagent --play --agent-file monte_carlo.pickle --episodes 200 > log.txt`

`uv run python -m ceo.cli.play_qagent --play --episodes 100000 --agent-file monte_carlo.pickle`

`uv run python -m ceo.cli.play_qagent --play --ppo-file eval_log/PPOM01/best_model.zip --episodes 200 > log.txt`

`uv run python -m ceo.cli.play_qagent --play-round-file play_hands/hands9.pickle --agent-file monte_carlo.pickle --do-logging > log.txt`

## Evaluate agents
`uv run python -m ceo.cli.eval_agents --num-players 6 --num-rounds 1000 --device cuda:1 --ppo-agents eval_log/BL_0_6_A eval_log/BL_1_6_A eval_log/BL_2_6_A eval_log/BL_3_6_A eval_log/BL_4_6_A eval_log/BL_5_6_A`

`uv run python -m ceo.cli.eval_agents --num-players 6 --num-rounds 1000 --device cpu --ppo-agents eval_log/BL_2_6_A eval_log/BL_3_6_A eval_log/BL_4_6_A eval_log/BL_5_6_A --basic-agent-seats 0 1`


## Azure command-line client

`uv run python -m ceo.cli.azure_rl --get-rl-trainings`

`uv run python -m ceo.cli.azure_rl --get-blob blob_name --save-file save.txt`

`uv run python -m ceo.cli.azure_rl --get-training-progress notebooks/progress.pkl --earliest-start 2022-03-10T00:00:00`

## Azure administration

`uv run python azure_admin.py <options>`

`uv run python azure_admin.py --train ../data/experiment_2022-02-12.json`
