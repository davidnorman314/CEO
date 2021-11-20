import gym
import numpy as np
import pickle
import datetime
from azure_rl.azure_client import AzureClient

from gym_ceo.envs.seat_ceo_env import CEOActionSpace
from gym_ceo.envs.actions import ActionEnum

from multiprocessing import RawArray


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
    _state_count: np.ndarray
    _max_action_value: int

    q_raw_array: RawArray
    state_count_raw_array: RawArray

    def __init__(self, env: gym.Env, **kwargs):
        """Initialize the Q table.
        If kwargs is empty, create normal np.ndarrays
        If kwargs has shared=True, then create the np.ndarrays using RawArrays
        so that they can be shared across processes.
        If kwargs has shared_q=RawArray and shared_state_count=RawArray, then create
        the ndarrays from the passed RawArrays.
        If kwargs has q and state_count defined, then their values must be ndarrays
        and they will be used for the arrays.
        """
        self._max_action_value = env.max_action_value

        # Extract the space
        obs_space = env.observation_space
        obs_shape = obs_space.shape
        assert len(obs_shape) == 1

        print("Observation space", obs_space)
        print("Observation space shape", obs_shape)
        print("Action space", env.action_space)

        # Calculate the shape of the arrays
        q_dims = ()
        for dim in obs_space.high:
            q_dims = q_dims + (dim + 1,)
        q_dims = q_dims + (env.max_action_value,)
        q_size = int(np.prod(q_dims))

        q_type = np.float32
        state_count_type = np.int32

        q_ctype = np.ctypeslib.as_ctypes_type(q_type)
        state_count_ctype = np.ctypeslib.as_ctypes_type(state_count_type)

        if not kwargs:
            # Initialize the arrays as np.ndarrys
            self._Q = np.zeros(q_dims, dtype=q_type)
            self._state_count = np.zeros(q_dims, dtype=state_count_type)
            print("Q dims", q_dims)
            print("Q table size", self._Q.nbytes // (1024 * 1024), "mb")

            init_from_shared = False
        elif "shared" in kwargs and kwargs["shared"]:
            # Initialize the arrays using shared RawArrays.
            print("q_ctype", q_ctype)
            print("q_size", q_size)
            self.q_raw_array = RawArray(q_ctype, q_size)
            self.state_count_raw_array = RawArray(state_count_ctype, q_size)

            init_from_shared = True
        elif "shared_q" in kwargs and "shared_state_count" in kwargs:
            # Initialize from shared RawArrays
            self.q_raw_array = kwargs["shared_q"]
            self.state_count_raw_array = kwargs["shared_state_count"]

            init_from_shared = True
        elif "q" in kwargs and "state_count" in kwargs:
            # Initialize from the passed arrays
            self._Q = kwargs["q"]
            self._state_count = kwargs["state_count"]

            init_from_shared = False
        else:
            raise Exception("Incorrect args in QTable constructor")

        if init_from_shared:
            self._Q = np.frombuffer(self.q_raw_array, dtype=q_type).reshape(q_dims)
            self._state_count = np.frombuffer(
                self.state_count_raw_array, dtype=state_count_type
            ).reshape(q_dims)

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

    def get_shared_arrays(self):
        return self.q_raw_array, self.state_count_raw_array


class LearningBase:
    """
    Base class for classes that learn how to play CEO well.
    """

    _env: gym.Env
    _qtable: QTable

    _search_statistics: list[dict]
    _start_time: datetime.datetime
    _last_backup_pickle_time: datetime.datetime
    _last_azure_log_time: datetime.datetime

    _azure_client: AzureClient

    def __init__(self, env: gym.Env, **kwargs):
        """Constructor for a learning base class object.
        The kwargs are passed to the QTable constructor so it can be initialized
        for multiprocessing.
        """
        if "azure_client" in kwargs:
            self._azure_client = kwargs["azure_client"]
            del kwargs["azure_client"]
        else:
            self._azure_client = None

        self._env = env
        self._qtable = QTable(env, **kwargs)

        self._search_statistics = []
        self._start_time = datetime.datetime.now()
        self._last_backup_pickle_time = datetime.datetime.now()
        self._last_azure_log_time = datetime.datetime.now()

    def set_env(self, env: gym.Env):
        """Sets the environment used by the agent"""
        self._env = env

    def add_search_statistics(
        self,
        typestr: str,
        episode: int,
        avg_reward: float,
        recent_reward: float,
        explore_rate: float,
        states_visited: int,
    ):
        now = datetime.datetime.now()

        stats = dict()
        stats["episode"] = episode
        stats["avg_reward"] = avg_reward
        stats["recent_reward"] = recent_reward
        stats["explore_rate"] = explore_rate
        stats["states_visited"] = states_visited
        stats["duration"] = now - self._start_time

        self._search_statistics.append(stats)

        if now - self._last_backup_pickle_time > datetime.timedelta(minutes=15):
            print("Pickling backup")
            self.pickle(typestr, "searchbackup.pickle")

            self._last_backup_pickle_time = now

        if self._azure_client:
            # if now - self._last_azure_log_time > datetime.timedelta(seconds=10):
            if now - self._last_azure_log_time > datetime.timedelta(minutes=5):
                self._azure_client.log(stats)

                self._last_azure_log_time = now

    def pickle(self, typestr: str, filename: str):
        pickle_dict = dict()
        pickle_dict["Q"] = self._qtable._Q
        pickle_dict["StateCount"] = self._qtable._state_count
        pickle_dict["Type"] = typestr
        pickle_dict["MaxActionValue"] = self._qtable._max_action_value
        pickle_dict["SearchStats"] = self._search_statistics

        with open(filename, "wb") as f:
            pickle.dump(pickle_dict, f, pickle.HIGHEST_PROTOCOL)

        # Upload to Azure, if necessary
        if self._azure_client:
            self._azure_client.upload_pickle(filename)

    def mean_squared_difference(self, o) -> int:
        """
        Calculates the mean squared difference between this QTable and the
        passed QTable.
        """

        return np.square(self._Q - o).mean(axis=None)
