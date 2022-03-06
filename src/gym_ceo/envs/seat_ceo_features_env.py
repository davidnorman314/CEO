import gym
import numpy as np
from gym.spaces import Box, Discrete

from gym_ceo.envs.seat_ceo_env import SeatCEOEnv
from gym_ceo.envs.actions import Actions
from gym_ceo.envs.observation import ObservationFactory, Observation
from gym_ceo.envs.features import *

from CEO.cards.round import Round, RoundState
from CEO.cards.eventlistener import EventListenerInterface
from CEO.cards.deck import Deck
from CEO.cards.hand import Hand, CardValue
from CEO.cards.simplebehavior import BasicBehavior
from CEO.cards.player import Player


class SeatCEOFeaturesEnv(gym.Env):
    """
    Environment for a player in the CEO seat. This environment reduces the observation space
    to a set of features.
    """

    metadata = {"render.modes": ["human"]}

    full_env: SeatCEOEnv

    observation_space: Box
    action_space: Discrete
    max_action_value: int

    # The number of players
    num_players: int

    _observation_factory: FeatureObservationFactory

    _observation_dimension: int

    def __init__(self, full_env: SeatCEOEnv, *, feature_defs=None):
        self.full_env = full_env
        self.action_space = full_env.action_space
        self.max_action_value = full_env.max_action_value
        self.num_players = full_env.num_players

        # Get feature definitions
        if feature_defs is None:
            print("Using default features")
            feature_defs = self.get_default_features(full_env)

        self._observation_factory = FeatureObservationFactory(full_env, feature_defs)

        self.observation_space = self._observation_factory.observation_space
        self._observation_dimension = self._observation_factory.observation_dimension

        print(
            "SeatCEOFeatures env has",
            self._observation_dimension,
            "dimensional observation space with ",
            self._observation_factory.obs_space_possible_values,
            "possible values",
        )

    def get_default_features(self, full_env: SeatCEOEnv):
        self.feature_defs = []
        half_players = full_env.num_players // 2
        for i in range(half_players - 1):
            feature_params = dict()
            feature_params["other_player_index"] = i
            feature_params["max_value"] = 5
            self.feature_defs.append(("OtherPlayerHandCount", feature_params))

        if False:
            feature_params = dict()
            self.feature_defs.append(("BottomHalfTableMinCards", feature_params))

        min_card_exact_feature = 9
        for i in range(min_card_exact_feature, 13):
            feature_params = dict()
            feature_params["card_value_index"] = i
            self.feature_defs.append(("HandCardCount", feature_params))

        feature_params = dict()
        feature_params["threshold"] = min_card_exact_feature
        self.feature_defs.append(("SinglesUnderValueCount", feature_params))

        feature_params = dict()
        feature_params["threshold"] = min_card_exact_feature
        self.feature_defs.append(("DoublesUnderValueCount", feature_params))

        feature_params = dict()
        feature_params["threshold"] = min_card_exact_feature
        self.feature_defs.append(("TriplesUnderValueCount", feature_params))

        feature_params = dict()
        self.feature_defs.append(("TrickPosition", feature_params))

        feature_params = dict()
        self.feature_defs.append(("CurTrickValue", feature_params))

        feature_params = dict()
        self.feature_defs.append(("CurTrickCount", feature_params))

        return self.feature_defs

    def reset(self, hands: list[Hand] = None):
        full_obs = self.full_env.reset(hands)
        self.action_space = self.full_env.action_space

        info = dict()
        return self.make_feature_observation(full_obs, info)

    def step(self, action):
        full_obs, reward, done, info = self.full_env.step(action)
        self.action_space = self.full_env.action_space

        obs = self.make_feature_observation(full_obs, info)
        return obs, reward, done, info

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def make_feature_observation(self, full_obs_array, info: dict):
        return self._observation_factory.make_feature_observation(full_obs_array, info)
