# Reinforcement learning for the card game CEO

## Setup
`pip3 install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu117`

## Run a specific test
`pytest -s -k game`

## Play games from the command line
`python -m main.console_game`

`python -m main.many_games`

`python -m main.many_games --print --count 1`

`python -m main.console_game --play-ceo --agent-file training.pkl`

`python -m main.console_game --play-ceo --ppo-file eval_log/PPOM01/best_model.zip --device cpu`


## Train agents

`python -m learning.qlearning --episodes 1000000`

`python -m learning.qlearning_afterstates --episodes 10000000`

`python -m learning.learning --episodes 10000000 --during-training-stats-episodes 1000 --during-training-stats-frequency 50000`

`python -m learning.qlearning_traces --episodes 100000`

`python -m learning.monte_carlo --train --episodes 1000000`

`python -m learning.monte_carlo --train --episodes 5000 --processes 3`

`python -m learning.learning --pickle-file training.pkl --disable-agent-testing ../data/qlearning.json`

`python -m learning.learning --pickle-file training.pkl --disable-agent-testing ../data/qlearning_traces_features.json`

`python -m learning.ppo --name PPOTest --n-steps-per-update 64 --batch-size 64 --learning-rate 3e-5 --pi-net-arch "64 64" --vf-net-arch "64 64" --device cpu`

`python -m learning.ppo --name PPOTest --n-steps-per-update 64 --batch-size 64 --learning-rate 3e-5 --pi-net-arch "64 64" --vf-net-arch "64 64" --device cpu --ppo-agents eval_log/BL_0_6_A`

### Continue training
`python -m learning.ppo --continue-training --name PPOTest --device cpu --ppo-agents eval_log/BL_0_6_A`

## Use trained agents to play

`python -m learning.play_qagent --play --episodes 100 --ppo-dir eval_log/PPOM01 --device cpu`

`python -m learning.play_qagent --play --agent-file monte_carlo.pickle --episodes 200 > log.txt`

`python -m learning.play_qagent --play --episodes 100000 --agent-file monte_carlo.pickle`

`python -m learning.play_qagent --play --ppo-file eval_log/PPOM01/best_model.zip --episodes 200 > log.txt`

`python -m learning.play_qagent --play-round-file play_hands/hands9.pickle --agent-file monte_carlo.pickle --do-logging > log.txt`

## Evaluate agents
`python -m learning.eval_agents --num-players 6 --num-rounds 1000 --device cuda:1 --ppo-agents eval_log/BL_0_6_A eval_log/BL_1_6_A eval_log/BL_2_6_A eval_log/BL_3_6_A eval_log/BL_4_6_A eval_log/BL_5_6_A`

`python -m learning.eval_agents --num-players 6 --num-rounds 1000 --device cpu --ppo-agents eval_log/BL_2_6_A eval_log/BL_3_6_A eval_log/BL_4_6_A eval_log/BL_5_6_A --basic-agent-seats 0 1`


## Azure command-line client

`python -m main.azure_rl --get-rl-trainings`

`python -m main.azure_rl --get-blob blob_name --save-file save.txt`

`python -m main.azure_rl --get-training-progress ../notebooks/progress.pkl --earliest-start 2022-03-10T00:00:00`

## Azure administation

`python azure_admin.py <options>`

`python azure_admin.py --train ../data/experiment_2022-02-12.json`
