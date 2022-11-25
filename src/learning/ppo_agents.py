"""Agents that use trained PPO models to play CEO."""

import pathlib
import json
import numpy as np
import torch as th

from CEO.cards.hand import Hand, CardValue
from CEO.cards.player import PlayerBehaviorInterface
from CEO.cards.simplebehavior import SimpleBehaviorBase
from CEO.cards.round import RoundState

from gym_ceo.envs.observation import Observation, ObservationFactory

from stable_baselines3 import PPO

import gym


def load_ppo(ppo_dir: str, device: str) -> tuple[PPO, dict]:
    if device is None:
        raise Exception("No device specified.")

    ppo = PPO.load(pathlib.Path(ppo_dir, "best_model.zip"), device=device, print_system_info=True)

    with open(pathlib.Path(ppo_dir, "params.json"), "rb") as f:
        params = json.load(f)

    return ppo, params


class PPOBehavior(PlayerBehaviorInterface, SimpleBehaviorBase):
    """Class that implements a player behavior using a trained PPO agent."""

    _seat_num: int
    _num_players: int
    _ppo: PPO
    _observation_factory: ObservationFactory

    source: str
    """The directory the ppo agent was loaded from."""

    def __init__(self, seat_num: int, num_players: int, ppo: PPO, params: dict, device: str):
        self.is_reinforcement_learning = False
        self._seat_num = seat_num
        self._num_players = num_players
        self._ppo = ppo

        obs_kwargs = params["env_args"]["obs_kwargs"]

        assert params["env_args"]["action_space_type"] == "all_card"
        assert params["env_args"]["num_players"] == num_players
        assert params["env_args"]["seat_number"] == seat_num

        self._observation_factory = ObservationFactory(
            seat_number=seat_num, num_players=num_players, **obs_kwargs
        )

    def pass_cards(self, hand: Hand, count: int) -> list[CardValue]:
        # Use the default pass method.
        return self.pass_singles(hand, count)

    def _get_action(self, obs):
        action_array, _ = self._ppo.predict(obs, deterministic=True)
        action = int(action_array)

        if action < 13:
            return CardValue(action)
        else:
            return None

    def lead(self, player_position: int, hand: Hand, state: RoundState) -> CardValue:
        obs = self._observation_factory.create_observation(
            type="lead", starting_player=player_position, cur_hand=hand, state=state
        )

        return self._get_action(obs)

    def play_on_trick(
        self,
        starting_position: int,
        player_position: int,
        hand: Hand,
        cur_trick_value: CardValue,
        cur_trick_count: int,
        state: RoundState,
    ) -> CardValue:
        obs = self._observation_factory.create_observation(
            type="play",
            cur_index=player_position,
            starting_position=starting_position,
            cur_card_value=cur_trick_value,
            cur_card_count=cur_trick_count,
            cur_hand=hand,
            state=state,
        )

        return self._get_action(obs)


class PPOAgent:
    """Class that uses a trained PPO agent to play a CEO game given by a gym environment."""

    _env: gym.Env

    _ppo: PPO

    _device: str

    def __init__(self, env: gym.Env, ppo: PPO, device: str):
        self._env = env
        self._ppo = ppo
        self._device = device

    def do_episode(
        self, hands: list[Hand] = None, log_state: bool = False
    ) -> tuple[list[tuple], list[int], float]:
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
