import gym
import gym.spaces
import numpy as np
from multiprocessing import RawArray


class ValueTable:
    """Class implementing a table giving the estimated expected value for each state."""

    _V: np.ndarray
    _state_count: np.ndarray

    _denom = 32000
    """We store value table values as integers. The actual value is the integer
    divided by _denom."""

    _max_state_visit = 63000

    v_raw_array: RawArray
    state_count_raw_array: RawArray

    def __init__(self, observation_space: gym.spaces.Box, **kwargs):
        """Initialize the value table.
        If kwargs is empty, create normal np.ndarrays
        If kwargs has shared=True, then create the np.ndarrays using RawArrays
        so that they can be shared across processes.
        If kwargs has shared_q=RawArray and shared_state_count=RawArray, then create
        the ndarrays from the passed RawArrays.
        If kwargs has v and state_count defined, then their values must be ndarrays
        and they will be used for the arrays.
        """

        # Extract the space
        obs_shape = observation_space.shape
        assert len(obs_shape) == 1

        print("Observation space", observation_space)
        print("Observation space shape", obs_shape)

        # Calculate the shape of the arrays
        v_dims = ()
        for dim in observation_space.high:
            v_dims = v_dims + (dim + 1,)
        v_size = int(np.prod(v_dims))

        v_type = np.int16
        state_count_type = np.uint16

        v_type_iinfo = np.iinfo(v_type)
        state_count_type_iinfo = np.iinfo(state_count_type)

        assert v_type_iinfo.max > self._denom
        assert v_type_iinfo.min < -self._denom

        assert state_count_type_iinfo.max > self._max_state_visit
        assert state_count_type_iinfo.min == 0

        print("v_type:", v_type, "[", v_type_iinfo.min, ",", v_type_iinfo.max, "]")
        print(
            "state_count_type:",
            state_count_type,
            "[",
            state_count_type_iinfo.min,
            ",",
            state_count_type_iinfo.max,
            "]",
        )

        v_ctype = np.ctypeslib.as_ctypes_type(v_type)
        state_count_ctype = np.ctypeslib.as_ctypes_type(state_count_type)

        if not kwargs:
            # Initialize the arrays as np.ndarrys
            self._V = np.zeros(v_dims, dtype=v_type)
            self._state_count = np.zeros(v_dims, dtype=state_count_type)
            print("Q dims", v_dims)
            print("Q table size", self._V.nbytes // (1024 * 1024), "mb")
            print("State count size", self._state_count.nbytes // (1024 * 1024), "mb")

            init_from_shared = False
        elif "shared" in kwargs and kwargs["shared"]:
            # Initialize the arrays using shared RawArrays.
            print("q_ctype", v_ctype)
            print("q_size", v_size)
            self.q_raw_array = RawArray(v_ctype, v_size)
            self.state_count_raw_array = RawArray(state_count_ctype, v_size)

            init_from_shared = True
        elif "shared_q" in kwargs and "shared_state_count" in kwargs:
            # Initialize from shared RawArrays
            self.q_raw_array = kwargs["shared_q"]
            self.state_count_raw_array = kwargs["shared_state_count"]

            init_from_shared = True
        elif "q" in kwargs and "state_count" in kwargs:
            # Initialize from the passed arrays
            self._V = kwargs["q"]
            self._state_count = kwargs["state_count"]

            init_from_shared = False
        else:
            raise Exception("Incorrect args in QTable constructor")

        if init_from_shared:
            self._V = np.frombuffer(self.q_raw_array, dtype=v_type).reshape(v_dims)
            self._state_count = np.frombuffer(
                self.state_count_raw_array, dtype=state_count_type
            ).reshape(v_dims)

    def state_visit_count(self, state_tuple: tuple):
        return self._state_count[state_tuple]

    def state_value(self, state_tuple: tuple):
        return self._V[state_tuple] / self._denom

    def increment_state_visit_count(self, state_tuple: tuple):
        self._state_count[state_tuple] += 1

        if self._state_count[state_tuple] > self._max_state_visit:
            self._state_count[state_tuple] = self._max_state_visit

    def update_state_value(self, state_tuple: tuple, delta: float):
        delta_int = delta * self._denom
        assert delta_int != 0 or delta == 0
        self._V[state_tuple] += delta_int
        if False:
            before = self._V[state_tuple] / self._denom
            after = before + delta
            self._V[state_tuple] = after / self._denom

    def get_shared_arrays(self):
        return self.q_raw_array, self.state_count_raw_array
