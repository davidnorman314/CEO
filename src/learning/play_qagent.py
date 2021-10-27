"""Play rounds using an agent based on a Q table
"""

import gym
import random
import pickle
import argparse
from typing import List, Tuple
from copy import copy, deepcopy
import numpy as np
from learning.learning_base import LearningBase
from collections import deque

from gym_ceo.envs.seat_ceo_env import SeatCEOEnv
from gym_ceo.envs.seat_ceo_features_env import SeatCEOFeaturesEnv
from CEO.cards.eventlistener import EventListenerInterface, GameWatchListener, PrintAllEventListener
from CEO.cards.deck import Deck
from CEO.cards.hand import Hand, CardValue


class QAgent(LearningBase):
    _base_env: gym.Env

    def __init__(self, q: np.ndarray, state_count: np.ndarray, env: gym.Env, base_env: gym.Env):
        super().__init__(env)

        self._base_env = base_env

        self._Q = q
        self._state_count = state_count

    def _pick_action(self, state_tuple: tuple):
        # The number of times we have visited this state
        n_state = np.sum(self._state_count[(*state_tuple, slice(None))])
        max_value = np.max(self._Q[(*state_tuple, slice(None))])
        min_value = np.min(self._Q[(*state_tuple, slice(None))])

        # Do the greedy action
        action = np.argmax(self._Q[(*state_tuple, slice(None))])

        # Clip the action, if necessary. This biases the exploration
        # toward leading the lowest card.
        if action >= self._env.action_space.n:
            action = self._env.action_space.n - 1

        return action

    def do_episode(
        self, hands: list[Hand] = None, log_state: bool = False
    ) -> Tuple[List[tuple], List[int], float]:
        """Plays a hand. Returns a list of states visited, actions taken, and the reward"""
        # Reseting the environment each time as per requirement
        state = self._env.reset(hands)
        state_tuple = tuple(state.astype(int))

        episode_states = []
        episode_actions = []
        episode_reward = 0

        # Run until the episode is finished
        while True:
            action = self._pick_action(state_tuple)

            if log_state:
                print("State", state_tuple)
                print("Action", action)

                for a in range(self._base_env.max_action_value):
                    print(
                        "  action",
                        a,
                        "value",
                        self._Q[(*state_tuple, a)],
                        "count",
                        self._state_count[(*state_tuple, a)],
                    )

            state_action_tuple = state_tuple + (action,)

            # Perform the action
            new_state, reward, done, info = self._env.step(action)

            episode_states.append(state_tuple)
            episode_actions.append(action)

            # print("state", state)
            # print("state_tuple", state_tuple)
            # print("action", action)
            # print("done", done)

            if new_state is not None:
                new_state_tuple = tuple(new_state.astype(int))
                new_state_value = np.max(self._Q[(*new_state_tuple, slice(None))])
            else:
                assert done
                assert reward != 0

                new_state_tuple = None
                new_state_value = 0

            # print("hand 2", info["hand"])
            # print("New q", type(self._Q[state_action_tuple]))
            # print("Q shape", self._Q.shape)
            # print("State len", len(state))
            # print("State shape", state.shape)

            # Increasing our total reward and updating the state
            episode_reward += reward
            state = new_state
            state_tuple = new_state_tuple

            # See if the episode is finished
            if done == True:
                # print("Reward", reward)
                break

        return episode_states, episode_actions, episode_reward


def create_agent(file_name: str, env: gym.Env, base_env: gym.Env):
    with open(file_name, "rb") as f:
        info = pickle.load(f)

    return QAgent(info["Q"], info["StateCount"], env, base_env)


def play(agent_file_name: str, episodes: int):

    # Set up the environment
    random.seed(0)

    listener = PrintAllEventListener()
    listener = GameWatchListener("RL")
    base_env = SeatCEOEnv(listener=listener)
    env = SeatCEOFeaturesEnv(base_env)

    # Load the agent
    agent = create_agent(agent_file_name, env, base_env)

    # Play the episodes
    total_wins = 0
    total_losses = 0
    for count in range(episodes):
        print("Playing episode", count + 1)
        deck = Deck(base_env.num_players)
        hands = deck.deal()
        save_hands = deepcopy(hands)

        states, actions, reward = agent.do_episode(hands, True)

        if reward > 0.0:
            total_wins += 1
        else:
            total_losses += 1

        if reward < 0:
            file = "play_hands/hands" + str(count + 1) + ".pickle"
            with open(file, "wb") as f:
                pickle.dump(save_hands, f, pickle.HIGHEST_PROTOCOL)

            for i in range(len(states)):
                print(i, "state", states[i], "action", actions[i])

    pct_win = total_wins / (total_wins + total_losses)
    print(
        "Episodes",
        episodes,
        "Total wins",
        total_wins,
        "Total losses",
        total_losses,
        "Percent wins",
        pct_win,
    )


def play_round(agent_file_name: str, round_pickle_file: str):
    # Load the hands
    with open(round_pickle_file, "rb") as f:
        hands = pickle.load(f)

    # Set up the environment
    random.seed(0)

    listener = PrintAllEventListener()
    listener = GameWatchListener("RL")
    base_env = SeatCEOEnv(listener=listener)
    env = SeatCEOFeaturesEnv(base_env)

    # Load the agent
    agent = create_agent(agent_file_name, env, base_env)

    states, actions, reward = agent.do_episode(hands, True)

    for i in range(len(states)):
        print(i, "state", states[i], "action", actions[i])

        for a in range(base_env.max_action_value):
            print(
                "  action",
                a,
                "value",
                agent._Q[(*states[i], a)],
                "count",
                agent._state_count[(*states[i], a)],
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        dest="profile",
        action="store_const",
        const=True,
        default=False,
        help="Do profiling.",
    )

    parser.add_argument(
        "--episodes",
        dest="episodes",
        type=int,
        default=100000,
        help="The number of rounds to play",
    )

    parser.add_argument(
        "--agent-file",
        dest="agent_file",
        type=str,
        help="The pickle file containing the agent",
    )

    parser.add_argument(
        "--play",
        dest="play",
        action="store_const",
        const=True,
        help="Have a trained agent play games",
    )

    parser.add_argument(
        "--play-round-file",
        dest="play_round_file",
        type=str,
        default=None,
        help="The name of a pickle file containing a list of hands",
    )

    args = parser.parse_args()
    # print(args)

    if args.play_round_file:
        play_round(args.agent_file, args.play_round_file)
    elif args.play:
        play(args.agent_file, args.episodes)
    else:
        parser.print_usage()


if __name__ == "__main__":
    # execute only if run as a script
    main()
