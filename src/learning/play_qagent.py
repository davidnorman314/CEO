"""Play rounds using an agent based on a Q table.
"""

import gym
import random
import pickle
import argparse
from typing import List, Tuple
from copy import copy, deepcopy
import numpy as np
from collections import deque

from gym_ceo.envs.seat_ceo_env import SeatCEOEnv, CEOActionSpace
from gym_ceo.envs.seat_ceo_features_env import SeatCEOFeaturesEnv
from gym_ceo.envs.actions import ActionEnum
from azure_rl.azure_client import AzureClient
from CEO.cards.eventlistener import (
    EventListenerInterface,
    GameWatchListener,
    PrintAllEventListener,
)
from CEO.cards.deck import Deck
from CEO.cards.hand import Hand, CardValue
from learning.learning_base import QTable


class QAgent:
    _base_env: gym.Env
    _env: gym.Env

    _qtable: QTable

    def __init__(self, q: np.ndarray, state_count: np.ndarray, env: gym.Env, base_env: gym.Env):
        self._base_env = base_env
        self._env = env
        self._qtable = QTable(env, q=q, state_count=state_count)

    def _pick_action(self, state_tuple: tuple):
        # Do the greedy action
        return self._qtable.greedy_action(state_tuple, self._env.action_space)

    def do_episode(
        self, hands: list[Hand] = None, log_state: bool = False
    ) -> Tuple[List[tuple], List[int], float]:
        """Plays a hand. Returns a list of states visited, actions taken, and the reward"""
        # Reseting the environment each time as per requirement
        state = self._env.reset(hands)
        info = dict()
        state_tuple = tuple(state.astype(int))

        episode_states = []
        episode_actions = []
        episode_reward = 0

        # Run until the episode is finished
        while True:
            selected_action = self._pick_action(state_tuple)

            if log_state:
                action_space = self._env.action_space

                print("State", state_tuple)
                print("Obs info:")
                for key, value in info.items():
                    print(" ", key, "->", value)

                print("Selected action", selected_action)
                print("Action values")
                for a in range(self._base_env.max_action_value):
                    full_action = ActionEnum(a)
                    name = full_action.name if full_action in action_space.actions else ""
                    selected = "selected" if full_action == selected_action else ""

                    print(
                        "  action",
                        a,
                        "value",
                        self._qtable.state_action_value((*state_tuple, full_action)),
                        "count",
                        self._qtable.state_visit_count((*state_tuple, full_action)),
                        name,
                    )

            state_action_tuple = state_tuple + (selected_action,)

            # Perform the action
            selected_action_index = self._env.action_space.actions.index(selected_action)
            new_state, reward, done, info = self._env.step(selected_action_index)

            episode_states.append(state_tuple)
            episode_actions.append(selected_action)

            # print("state", state)
            # print("state_tuple", state_tuple)
            # print("action", action)
            # print("done", done)

            if new_state is not None:
                new_state_tuple = tuple(new_state.astype(int))
            else:
                assert done
                assert reward != 0

                new_state_tuple = None

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


def create_agent(
    listener: EventListenerInterface,
    *,
    local_file=None,
    azure_blob_name=None,
    env=None,
    base_env=None,
    q_table=None,
    state_count=None,
):

    if local_file or azure_blob_name:
        assert not base_env
        assert not env
        assert not q_table
        assert not state_count

        # Load the pickle file and use it to create the environment and agent.
        if local_file:
            with open(local_file, "rb") as f:
                info = pickle.load(f)
        elif azure_blob_name:
            print("Downloading from Azure")
            client = AzureClient()
            blob = client.get_blob_raw(azure_blob_name)
            info = pickle.loads(blob)
            print("Finished downloading from Azure")
        else:
            print("Error: No picked agent specified.")
            return

        if "FeatureDefs" in info:
            feature_defs = info["FeatureDefs"]
        else:
            feature_defs = None

        num_players = info["NumPlayers"]

        base_env = SeatCEOEnv(num_players=num_players, listener=listener)
        env = SeatCEOFeaturesEnv(base_env, feature_defs=feature_defs)

        q_table = info["Q"]
        state_count = info["StateCount"]

        print(info.keys())
        print("----")
        print("Trained with", info["SearchStats"][-1]["episode"], "episodes")
        print("----")
    elif env:
        assert not local_file
        assert not azure_blob_name

        assert q_table is not None
        assert state_count is not None
        assert env is not None
        assert base_env is not None
    else:
        print("Error: incorrect arguments specified.")

    return env, base_env, QAgent(q_table, state_count, env, base_env)


def play(episodes: int, do_logging: bool, save_failed_hands: bool, **kwargs):

    azure_client = None
    if "azure_client" in kwargs:
        azure_client = kwargs["azure_client"]
        del kwargs["azure_client"]

    # Set up the environment
    random.seed(0)

    if do_logging:
        listener = PrintAllEventListener()
        listener = GameWatchListener("RL")
    else:
        listener = EventListenerInterface()

    # Load the agent
    env, base_env, agent = create_agent(listener, **kwargs)

    # Play the episodes
    total_wins = 0
    total_losses = 0
    for count in range(episodes):
        if do_logging:
            print("Playing episode", count + 1)

        deck = Deck(base_env.num_players)
        hands = deck.deal()
        save_hands = deepcopy(hands)

        states, actions, reward = agent.do_episode(hands, do_logging)

        if reward > 0.0:
            total_wins += 1
        else:
            total_losses += 1

        if reward < 0:
            if save_failed_hands:
                file = "play_hands/hands" + str(count + 1) + ".pickle"
                with open(file, "wb") as f:
                    pickle.dump(save_hands, f, pickle.HIGHEST_PROTOCOL)

            if do_logging:
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

    if azure_client:
        azure_client.save_post_train_stats(
            episodes=episodes,
            total_wins=total_wins,
            total_losses=total_losses,
            pct_win=pct_win,
        )


def play_round(round_pickle_file: str, do_logging: bool, **kwargs):
    # Load the hands
    with open(round_pickle_file, "rb") as f:
        hands = pickle.load(f)

    # Set up the environment
    random.seed(0)

    if do_logging:
        listener = PrintAllEventListener()
        listener = GameWatchListener("RL")
    else:
        listener = EventListenerInterface()

    base_env = SeatCEOEnv(listener=listener)
    env = SeatCEOFeaturesEnv(base_env)

    # Load the agent
    env, base_env, agent = create_agent(listener, **kwargs)

    states, actions, reward = agent.do_episode(hands, do_logging)

    if False:
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
        "--azure-agent",
        dest="azure_agent",
        type=str,
        help="The name of the Auzre blob containing the pickled agent.",
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

    parser.add_argument(
        "--do-logging",
        dest="do_logging",
        action="store_const",
        const=True,
        default=False,
        help="Log information giving the details of each hand.",
    )

    parser.add_argument(
        "--save-failed-hands",
        dest="save_failed_hands",
        action="store_const",
        const=True,
        default=False,
        help="Save pickle files for hands where the agent got a negative reward.",
    )

    args = parser.parse_args()
    # print(args)

    agent_args = dict()
    if args.agent_file:
        agent_args["local_file"] = args.agent_file
    elif args.azure_agent:
        agent_args["azure_blob_name"] = args.azure_agent
    else:
        print("No agent file specified.")
        return

    if args.play_round_file:
        play_round(args.play_round_file, args.do_logging, **agent_args)
    elif args.play:
        play(args.episodes, args.do_logging, args.save_failed_hands, **agent_args)
    else:
        parser.print_usage()


if __name__ == "__main__":
    # execute only if run as a script
    main()
