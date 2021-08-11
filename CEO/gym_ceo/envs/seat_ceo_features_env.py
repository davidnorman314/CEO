import gym
import numpy as np
from gym import error, spaces, utils
from gym.spaces import Box, Discrete
from gym.utils import seeding

from gym_ceo.envs.seat_ceo_env import SeatCEOEnv
from gym_ceo.envs.actions import Actions

from CEO.cards.round import Round, RoundState
from CEO.cards.eventlistener import EventListenerInterface
from CEO.cards.deck import Deck
from CEO.cards.hand import Hand, CardValue
from CEO.cards.simplebehavior import BasicBehavior
from CEO.cards.player import Player


class BottomHalfMinCards:
    """
    Feature giving the minimum number of cards for players in the bottom half of the table.
    """

    dim = 1
    max_value = 5

    _start_check_index: int
    _end_check_index: int

    def __init__(self, full_env: SeatCEOEnv):
        self._start_check_index = (
            full_env.num_players // 2 + full_env.obs_index_other_player_card_count
        )
        self._end_check_index = full_env.num_players + full_env.obs_index_other_player_card_count

    def calc(self, full_obs: np.array, dest_obs: np.array, dest_start_index: int):
        feature_value = self.max_value
        for i in range(self._start_check_index, self._end_check_index):
            feature_value = min(feature_value, full_obs[i])

        dest_obs[dest_start_index] = feature_value


class SeatCEOFeaturesEnv(gym.Env):
    """
    Environment for a player in the CEO seat. This environment reduces the observation space
    to a set of features.
    """

    metadata = {"render.modes": ["human"]}

    full_env: SeatCEOEnv

    observation_space: Box
    action_space: Discrete

    # Objects to calculate the features
    _feature_calculators = []

    _observation_dimension: int

    def __init__(self, full_env: SeatCEOEnv):
        self.full_env = full_env
        self.action_space = full_env.action_space

        self._feature_calculators.append(BottomHalfMinCards(full_env))

        # Calculate the observation space
        obs_space_low = []
        obs_space_high = []
        for calculator in self._feature_calculators:
            for i in range(calculator.dim):
                obs_space_low.append(0)
                obs_space_high.append(calculator.max_value)
        self.observation_space = Box(
            low=np.array(obs_space_low),
            high=np.array(obs_space_high),
            dtype=np.int32,
        )

        self._observation_dimension = len(obs_space_high)

    def reset(self):
        full_obs = self.full_env.reset()
        self.action_space = self.full_env.action_space

        return self._make_observation(full_obs)

    def step(self, action):
        full_obs, reward, done, info = self.full_env.step(action)
        self.action_space = self.full_env.action_space

        return self._make_observation(full_obs), reward, done, info

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def _make_observation(self, full_obs):
        if full_obs is None:
            return None

        obs = np.zeros(self._observation_dimension)

        i = 0
        for calculator in self._feature_calculators:
            calculator.calc(full_obs, obs, i)
            i = i + calculator.dim

        return obs
