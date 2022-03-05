# Reinforcement learning for the card game CEO

## Run a specific test
pytest -s -k game

## Play games from the command line
`python -m main.console_game`

`python -m main.many_games`

`python -m main.many_games --print --count 1`

## Train agents

`python -m learning.qlearning --episodes 1000000`

`python -m learning.qlearning_traces --episodes 100000`

`python -m learning.monte_carlo --train --episodes 1000000`

`python -m learning.monte_carlo --train --episodes 5000 --processes 3`

`python -m learning.learning --pickle-file training.pkl --disable-agent-testing ../data/qlearning.json`

`python -m learning.learning --pickle-file training.pkl --disable-agent-testing ../data/qlearning_traces_features.json`

## Use trained agents to play

`python -m learning.play_qagent --play --agent-file monte_carlo.pickle --episodes 200 > log.txt`

`python -m learning.play_qagent --play-round-file play_hands/hands9.pickle --agent-file monte_carlo.pickle > log.txt`

## Azure command-line client

`python -m main.azure_rl --get-rl-trainings`

`python -m main.azure_rl --get-blob blob_name --save-file save.txt`

## Azure administation

`python azure_admin.py <options>`

`python azure_admin.py --train ../data/experiment_2022-02-12.json`
