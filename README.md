# Reinforcement learning for the card game CEO

## Run a specific test
pytest -s -k game

## Play games from the command line
`python -m main.console_game`

`python -m main.many_games`

`python -m main.many_games --print --count 1`

## Train agents

`python -m learning.qlearning --episodes 1000000`

`python -m learning.monte_carlo --train --episodes 1000000`

`python -m learning.monte_carlo --train --episodes 5000 --processes 3`

## Use trained agents to play

`python -m learning.play_qagent --play --agent-file monte_carlo.pickle --episodes 200 > log.txt`

`python -m learning.play_qagent --play-round-file play_hands/hands9.pickle --agent-file monte_carlo.pickle > log.txt`