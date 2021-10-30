import gym
import numpy as np
import pickle

from gym_ceo.envs.seat_ceo_env import CEOActionSpace
from gym_ceo.envs.actions import ActionEnum


class EpisodeInfo:
    state: np.ndarray
    state_visit_count: int
    alpha: float
    value_before: float
    value_after: float
    hand: object
    action_type: str


class QTable:
    _Q: np.ndarray
    _max_action_value: int
    _state_count: np.ndarray

    def __init__(self, env: gym.Env):
        self._max_action_value = env.max_action_value

        # Extract the space
        obs_space = env.observation_space
        obs_shape = obs_space.shape
        assert len(obs_shape) == 1

        print("Observation space", obs_space)
        print("Observation space shape", obs_shape)
        print("Action space", env.action_space)

        # Initialize the Q-table
        q_dims = ()
        for dim in obs_space.high:
            q_dims = q_dims + (dim + 1,)
        q_dims = q_dims + (env.max_action_value,)

        self._Q = np.zeros(q_dims, dtype=np.float32)
        self._state_count = np.zeros(q_dims, dtype=np.int32)
        print("Q dims", q_dims)
        print("Q table size", self._Q.nbytes // (1024 * 1024), "mb")

    def visit_count(self, state_tuple: tuple, action_space: CEOActionSpace):
        return sum(
            self._state_count[(*state_tuple, action.value)] for action in action_space.actions
        )
        # return np.sum(self._state_count[(*state_tuple, slice(None))])

    def state_visit_count(self, state_action_tuple: tuple):
        return self._state_count[(*state_action_tuple[:-1], state_action_tuple[-1].value)]

    def state_action_value(self, state_action_tuple: tuple):
        return self._Q[(*state_action_tuple[:-1], state_action_tuple[-1].value)]

    def min_max_value(self, state_tuple: tuple, action_space: CEOActionSpace):
        max_value = max(self._Q[(*state_tuple, action.value)] for action in action_space.actions)
        min_value = min(self._Q[(*state_tuple, action.value)] for action in action_space.actions)

        return (min_value, max_value)

    def greedy_action(self, state_tuple: tuple, action_space: CEOActionSpace) -> ActionEnum:
        lookup_value = lambda action: self._Q[(*state_tuple, action.value)]
        return max(action_space.actions, key=lookup_value)

    def state_value(self, state_tuple: tuple, action_space: CEOActionSpace):
        return max(self._Q[(*state_tuple, action.value)] for action in action_space.actions)

        lookup_value = lambda i: self._Q[(*state_tuple, i)]
        return max(range(action_space.n), key=lookup_value)

    def increment_state_visit_count(self, state_action_tuple: tuple):
        self._state_count[(*state_action_tuple[:-1], state_action_tuple[-1].value)] += 1

    def update_state_visit_value(self, state_action_tuple: tuple, delta: float):
        self._Q[(*state_action_tuple[:-1], state_action_tuple[-1].value)] += delta


class LearningBase:
    """
    Base class for classes that learn how to play CEO well.
    """

    _env: gym.Env
    _qtable: QTable

    def __init__(self, env: gym.Env):
        self._env = env
        self._qtable = QTable(env)

    def set_env(self, env: gym.Env):
        """Sets the environment used by the agent"""
        self._env = env

    def pickle(self, typestr: str, filename: str):
        pickle_dict = dict()
        pickle_dict["Q"] = self._qtable._Q
        pickle_dict["StateCount"] = self._qtable._state_count
        pickle_dict["Type"] = typestr
        pickle_dict["MaxActionValue"] = self._qtable._max_action_value

        with open(filename, "wb") as f:
            pickle.dump(pickle_dict, f, pickle.HIGHEST_PROTOCOL)

    def mean_squared_difference(self, o) -> int:
        """
        Calculates the mean squared difference between this QTable and the
        passed QTable.
        """

        return np.square(self._Q - o).mean(axis=None)
