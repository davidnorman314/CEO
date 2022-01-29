import gym
import numpy as np
from multiprocessing import RawArray

from gym_ceo.envs.actions import ActionEnum
from gym_ceo.envs.seat_ceo_env import CEOActionSpace


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
