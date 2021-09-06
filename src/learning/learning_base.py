import gym
import numpy as np


class EpisodeInfo:
    state: np.ndarray
    state_visit_count: int
    alpha: float
    value_before: float
    value_after: float
    hand: object
    action_type: str


class LearningBase:
    """
    Base class for classes that learn how to play CEO well.
    """

    _env: gym.Env
    _Q: np.ndarray
    _max_action_value: int
    _state_count: np.ndarray

    def __init__(self, env: gym.Env):
        self._env = env
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

    def mean_squared_difference(self, o) -> int:
        """
        Calculates the mean squared difference between this QTable and the
        passed QTable.
        """

        return np.square(self._Q - o).mean(axis=None)
