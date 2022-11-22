"""Play rounds using an agent based on a Q table.
"""

import gym
import random
import pickle
import argparse
import json
import pathlib
from typing import List, Tuple
from copy import copy, deepcopy
import numpy as np
from statsmodels.stats.proportion import proportion_confint
from collections import deque
from typing import NamedTuple

from gym_ceo.envs.ceo_player_env import CEOPlayerEnv
from gym_ceo.envs.seat_ceo_env import SeatCEOEnv
from gym_ceo.envs.seat_ceo_features_env import SeatCEOFeaturesEnv
from gym_ceo.envs.features import FeatureObservationFactory
from gym_ceo.envs.actions import ActionEnum
from azure_rl.azure_client import AzureClient
from CEO.cards.eventlistener import (
    EventListenerInterface,
    GameWatchListener,
    PrintAllEventListener,
)
from CEO.cards.deck import Deck
from CEO.cards.hand import Hand, CardValue
from learning.qtable import QTable
from learning.value_table import ValueTable

from stable_baselines3 import PPO

import torch as th
import cloudpickle


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
    ) -> Tuple[List[tuple], List[int], float, dict]:
        """Plays a hand. Returns a list of states visited, actions taken, and the reward"""
        # Reseting the environment each time as per requirement
        state = self._env.reset(hands)
        info = dict()
        state_tuple = tuple(state.astype(int))

        episode_states = []
        episode_actions = []
        episode_reward = 0

        # Run until the episode is finished
        final_info = None
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
                final_info = info
                break

        return episode_states, episode_actions, episode_reward, final_info


class AfterstateAgent:
    _env: gym.Env
    _obs_factory: FeatureObservationFactory

    _valuetable: ValueTable

    def __init__(
        self, value_table: np.ndarray, state_count: np.ndarray, env: gym.Env, feature_defs
    ):
        self._env = env
        self._valuetable = ValueTable(env.observation_space, v=value_table, state_count=state_count)

        self._obs_factory = FeatureObservationFactory(env, feature_defs)

    def _pick_action(self, state: np.ndarray):
        action, expected_reward, visit_count = self._valuetable.find_greedy_action(
            self._env, self._obs_factory, state
        )

        return action, expected_reward, visit_count

    def do_episode(
        self, hands: list[Hand] = None, log_state: bool = False
    ) -> tuple[list[tuple], list[int], float, dict]:
        """Plays a hand. Returns a list of states visited, actions taken, and the reward"""
        state = self._env.reset(hands)
        info = dict()

        episode_states = []
        episode_actions = []
        episode_reward = 0

        # Run until the episode is finished
        final_info = None
        while True:
            selected_action, expected_reward, visit_count = self._pick_action(state)

            if log_state:
                print(
                    "selected action",
                    selected_action,
                    "expected reward",
                    expected_reward,
                    "visits",
                    visit_count,
                )
                # Needs to be fixed
                for action in range(self._env.action_space.n):
                    afterstate_observation, played_card = self._env.get_afterstate(state, action)
                    afterstate_feature_observation = self._obs_factory.make_feature_observation(
                        afterstate_observation, info
                    )

                    # Get the estimated expected reward for the afterstate
                    afterstate_tuple = tuple(afterstate_feature_observation.astype(int))

                    reward = self._valuetable.state_value(afterstate_tuple)
                    visit_count = self._valuetable.state_visit_count(afterstate_tuple)

                    full_action = self._env.action_space.actions[action]

                    print(
                        "action", action, "reward", reward, "visit count", visit_count, full_action
                    )

                    self._obs_factory.log_feature_observation(afterstate_feature_observation, "  ")

            # Perform the action
            new_state, reward, done, info = self._env.step(selected_action)

            episode_states.append(state)
            episode_actions.append(selected_action)

            if new_state is not None:
                pass
            else:
                assert done
                assert reward != 0

            # Increasing our total reward and updating the state
            episode_reward += reward
            state = new_state

            # See if the episode is finished
            if done == True:
                # print("Reward", reward)
                final_info = info
                break

        return episode_states, episode_actions, episode_reward, final_info


class PPOAgent:
    _env: gym.Env

    _ppo: PPO

    _device: str

    def __init__(self, env: gym.Env, ppo: PPO, device: str):
        self._env = env
        self._ppo = ppo
        self._device = device

    def do_episode(
        self, hands: list[Hand] = None, log_state: bool = False
    ) -> Tuple[List[tuple], List[int], float]:
        """Plays a hand. Returns a list of states visited, actions taken, and the reward"""
        # Reseting the environment each time as per requirement
        obs = self._env.reset(hands)
        info = dict()

        episode_states = []
        episode_actions = []
        episode_reward = 0

        # Run until the episode is finished
        final_info = None
        while True:
            selected_action_array, _ = self._ppo.predict(obs, deterministic=True)
            selected_action = int(selected_action_array)

            nparr = np.array([obs])
            obs_tensor_array = th.tensor(nparr, device=self._device)

            predicted_value = self._ppo.policy.predict_values(obs_tensor_array)[0][0]
            distribution = self._ppo.policy.get_distribution(obs_tensor_array).distribution.probs[0]

            if log_state:
                print("Obs", obs)
                print("Obs info:")
                for key, value in info.items():
                    print(" ", key, "->", value)
                print("Selected action", selected_action)

                print(f"Predicted value {predicted_value}")
                print(f"Action distribution {distribution}")

            # Perform the action
            new_obs, reward, done, info = self._env.step(selected_action)

            episode_states.append(f"value {predicted_value}")
            episode_actions.append(f"{selected_action} prob {distribution[selected_action]}")

            # Increasing our total reward and updating the state
            episode_reward += reward
            obs = new_obs

            # See if the episode is finished
            if done == True:
                # print("Reward", reward)
                final_info = info
                break

        return episode_states, episode_actions, episode_reward, final_info


def create_agent(
    listener: EventListenerInterface,
    *,
    local_file=None,
    ppo_dir=None,
    azure_blob_name=None,
    env=None,
    base_env=None,
    q_table=None,
    value_table=None,
    state_count=None,
    feature_defs=None,
    device=None,
):

    if ppo_dir:
        if device is None:
            raise Exception("No device specified.")

        print("Creating ppo agent")

        ppo = PPO.load(
            pathlib.Path(ppo_dir, "best_model.zip"), device=device, print_system_info=True
        )

        with open(pathlib.Path(ppo_dir, "params.json"), "rb") as f:
            params = json.load(f)

        env = CEOPlayerEnv(listener=listener, **params["env_args"])

        return env, env, PPOAgent(env, ppo, device)

    elif local_file or azure_blob_name:
        assert not base_env
        assert not env
        assert not q_table
        assert not value_table
        assert not state_count
        assert not feature_defs

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
            print(json.dumps(feature_defs, indent=2, separators=(",", ": ")))
        else:
            feature_defs = None

        num_players = info["NumPlayers"]

        if "Q" in info:
            # Create a QAgent
            base_env = SeatCEOEnv(num_players=num_players, listener=listener)
            env = SeatCEOFeaturesEnv(base_env, feature_defs=feature_defs)

            q_table = info["Q"]
            state_count = info["StateCount"]

            print(info.keys())
            print("----")
            print("Trained with", info["SearchStats"][-1]["episode"], "episodes")
            print("----")

            value_table = None
        elif "ValueTable" in info:
            # Create an afterstate agent
            env = SeatCEOEnv(num_players=num_players, listener=listener)

            value_table = info["ValueTable"]
            state_count = info["StateCount"]

            print(info.keys())
            print("----")
            print("Trained with", info["SearchStats"][-1]["episode"], "episodes")
            print("----")

            q_table = None
        else:
            print("Error: Pickled data does not have V or Q element: " + str(info.keys()))

    elif env:
        assert not local_file
        assert not azure_blob_name

        assert q_table is not None or value_table is not None
        assert state_count is not None
        assert env is not None
        assert base_env is not None or value_table is not None
    else:
        print("Error: incorrect arguments specified.")

    if q_table is not None:
        return env, base_env, QAgent(q_table, state_count, env, base_env)
    elif value_table is not None:
        assert feature_defs is not None
        return env, env, AfterstateAgent(value_table, state_count, env, feature_defs)


class PlayStats(NamedTuple):
    episodes: int
    total_wins: int
    total_losses: int
    pct_win: float


def play(episodes: int, do_logging: bool, save_failed_hands: bool, **kwargs) -> PlayStats:
    # Set up the environment
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
    total_stay = 0
    total_illegal_play = 0
    total_ceo_stay = 0
    total_ceo_to_bottom = 0
    for count in range(episodes):
        if do_logging:
            print("Playing episode", count + 1)
        elif count != 0 and count % 5000 == 0:
            print(f"Episode {count}")

        deck = Deck(base_env.num_players)
        hands = deck.deal()
        save_hands = deepcopy(hands)

        states, actions, reward, final_info = agent.do_episode(hands, do_logging)
        assert final_info is not None

        if reward > 0.0:
            total_wins += 1
        elif reward == -1.0:
            total_losses += 1
        elif reward == 0.0:
            total_stay += 1
        else:
            total_illegal_play += 1

        if reward < 0:
            if save_failed_hands:
                file = "play_hands/hands" + str(count + 1) + ".pickle"
                with open(file, "wb") as f:
                    pickle.dump(save_hands, f, pickle.HIGHEST_PROTOCOL)

            if do_logging:
                for i in range(len(states)):
                    print(i, "state", states[i], "action", actions[i])

        if "ceo_stay" in final_info:
            if final_info["ceo_stay"]:
                total_ceo_stay += 1
            else:
                total_ceo_to_bottom += 1

    pct_win = total_wins / episodes
    print(
        "Test episodes",
        episodes,
        "Total wins",
        total_wins,
        "Total stay",
        total_stay,
        "Total losses",
        total_losses,
        "Total illegal",
        total_illegal_play,
        "Percent wins",
        pct_win,
    )

    alpha = 0.05
    l, r = proportion_confint(count=total_wins, nobs=episodes, alpha=alpha, method="normal")
    print(f"{1-alpha} wins confidence interval ({l}, {r})")

    alpha = 0.01
    l, r = proportion_confint(count=total_wins, nobs=episodes, alpha=alpha, method="normal")
    print(f"{1-alpha} wins confidence interval ({l}, {r})")

    assert episodes == total_ceo_stay + total_ceo_to_bottom or (
        total_ceo_stay == 0 and total_ceo_to_bottom == 0
    )

    pct_ceo_stay = total_ceo_stay / episodes
    print(f"CEO stay {total_ceo_stay} CEO to bottom {total_ceo_to_bottom} pct stay {pct_ceo_stay}")

    return PlayStats(
        episodes=episodes,
        total_wins=total_wins,
        total_losses=total_losses,
        pct_win=pct_win,
    )


def play_round(hands: list[Hand], do_logging: bool, **kwargs):

    # Set up the environment
    if do_logging:
        listener = PrintAllEventListener()
        listener = GameWatchListener("RL")
    else:
        listener = EventListenerInterface()

    # Load the agent
    env, base_env, agent = create_agent(listener, **kwargs)

    states, actions, reward = agent.do_episode(hands, do_logging)

    if do_logging:
        for i in range(len(states)):
            print(i, "state", states[i], "action", actions[i])

            if hasattr(agent, "_Q"):
                for a in range(base_env.max_action_value):
                    print(
                        "  action",
                        a,
                        "value",
                        agent._Q[(*states[i], a)],
                        "count",
                        agent._state_count[(*states[i], a)],
                    )
        print("Reward", reward)

    return reward


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
        "--ppo-dir",
        dest="ppo_dir",
        type=str,
        help="The zip file containing the PPO agent. The directory should have best_model.zip and params.json.",
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
        "--device",
        dest="device",
        type=str,
        default=None,
        help="The name of the CUDA device to use.",
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

    parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=None,
        help="Set the random seed.",
    )

    args = parser.parse_args()

    agent_args = dict()
    if args.agent_file:
        agent_args["local_file"] = args.agent_file
    elif args.ppo_dir:
        agent_args["ppo_dir"] = args.ppo_dir
    elif args.azure_agent:
        agent_args["azure_blob_name"] = args.azure_agent
    else:
        print("No agent file specified.")
        return

    if args.device:
        agent_args["device"] = args.device

    if args.seed is not None:
        random.seed(args.seed)
    else:
        random.seed()

    if args.play_round_file:
        # Load the hands
        with open(args.play_round_file, "rb") as f:
            hands = pickle.load(f)

        play_round(hands, args.do_logging, **agent_args)
    elif args.play:
        stats = play(args.episodes, args.do_logging, args.save_failed_hands, **agent_args)
    else:
        parser.print_usage()


if __name__ == "__main__":
    # execute only if run as a script
    main()
