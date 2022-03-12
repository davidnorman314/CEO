import gym
import numpy as np
import pickle
import datetime
from azure_rl.azure_client import AzureClient

from learning.qtable import QTable
from learning.value_table import ValueTable
import learning.play_qagent as play_qagent


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

    _type: str
    _search_statistics: list[dict]
    _start_time: datetime.datetime
    _last_backup_pickle_time: datetime.datetime
    _last_azure_log_time: datetime.datetime

    _during_training_stats_episodes: int
    _during_training_stats_frequency: int

    _azure_client: AzureClient

    def __init__(self, type: str, env: gym.Env, **kwargs):
        """Constructor for a learning base class object."""
        if "azure_client" in kwargs:
            self._azure_client = kwargs["azure_client"]
            del kwargs["azure_client"]
        else:
            self._azure_client = None

        self._disable_agent_testing = kwargs["disable_agent_testing"]
        del kwargs["disable_agent_testing"]

        if "during_training_stats_episodes" in kwargs and kwargs["during_training_stats_episodes"]:
            self._during_training_stats_episodes = kwargs["during_training_stats_episodes"]
        else:
            self._during_training_stats_episodes = 100000
        del kwargs["during_training_stats_episodes"]

        if (
            "during_training_stats_frequency" in kwargs
            and kwargs["during_training_stats_frequency"]
        ):
            self._during_training_stats_frequency = kwargs["during_training_stats_frequency"]
        else:
            self._during_training_stats_frequency = 100000
        del kwargs["during_training_stats_frequency"]

        self._type = type
        self._env = env
        self._base_env = env

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

        if now - self._last_backup_pickle_time > datetime.timedelta(minutes=30):
            print("Pickling backup")
            self.pickle(typestr, "searchbackup.pickle")

            self._last_backup_pickle_time = now

        if self._azure_client:
            # if now - self._last_azure_log_time > datetime.timedelta(seconds=10):
            if now - self._last_azure_log_time > datetime.timedelta(minutes=5):
                self._azure_client.log(stats)

                self._last_azure_log_time = now

    def pickle(self, filename: str, feature_defs=None):
        pickle_dict = dict()
        pickle_dict["Type"] = self._type
        pickle_dict["SearchStats"] = self._search_statistics
        pickle_dict["NumPlayers"] = self._env.num_players

        if feature_defs is not None:
            pickle_dict["FeatureDefs"] = feature_defs
        else:
            pickle_dict["FeatureDefs"] = self._env.feature_defs

        self._init_pickle_dict(pickle_dict)

        data = pickle.dumps(pickle_dict, pickle.HIGHEST_PROTOCOL)

        if filename:
            print("Saving results to", filename)
            with open(filename, "wb") as f:
                f.write(data)

        # Upload to Azure, if necessary
        if self._azure_client:
            self._azure_client.upload_pickle(data=data)

    def _init_pickle_dict(self, pickle_dict: dict):
        raise NotImplementedError("Must be implemented in derived class")


class QTableLearningBase(LearningBase):
    _qtable: QTable

    def __init__(self, type: str, env: gym.Env, base_env: gym.Env, **kwargs):
        """Constructor for a learning base class object that uses a Q table.
        The kwargs are passed to the QTable constructor so it can be initialized
        for multiprocessing.
        """
        super().__init__(type, env, **kwargs)

        del kwargs["disable_agent_testing"]
        del kwargs["during_training_stats_episodes"]
        del kwargs["during_training_stats_frequency"]

        self._qtable = QTable(env, **kwargs)

    def _init_pickle_dict(self, pickle_dict: dict):
        pickle_dict["Q"] = self._qtable._Q
        pickle_dict["MaxActionValue"] = self._qtable._max_action_value
        pickle_dict["StateCount"] = self._qtable._state_count

    def do_play_test(self, training_episodes: int, **kwargs):
        """Plays many hands with the current agent and logs the results."""

        if self._disable_agent_testing:
            return

        q_table = self._qtable._Q
        state_count = self._qtable._state_count

        stats = play_qagent.play(
            self._during_training_stats_episodes,
            False,
            False,
            env=self._env,
            base_env=self._base_env,
            q_table=q_table,
            state_count=state_count,
            **kwargs
        )

        if self._azure_client:
            self._azure_client.save_test_stats(
                training_episodes=training_episodes,
                episodes=stats.episodes,
                total_wins=stats.total_wins,
                total_losses=stats.total_losses,
                pct_win=stats.pct_win,
            )

    def mean_squared_difference(self, o) -> int:
        """
        Calculates the mean squared difference between this QTable and the
        passed QTable.
        """

        return np.square(self._Q - o).mean(axis=None)


class ValueTableLearningBase(LearningBase):
    _valuetable: ValueTable

    def __init__(self, type: str, env: gym.Env, **kwargs):
        """Constructor for a learning base class object that uses a value table."""
        super().__init__(type, env, **kwargs)

        del kwargs["disable_agent_testing"]

        self._valuetable = None

    def _set_observation_space(self, observation_space, **kwargs):
        self._valuetable = ValueTable(observation_space, **kwargs)

    def _init_pickle_dict(self, pickle_dict: dict):
        pickle_dict["ValueTable"] = self._valuetable._V
        pickle_dict["StateCount"] = self._valuetable._state_count

    def do_play_test(self, training_episodes: int, feature_defs: list):
        """Plays many hands with the current agent and logs the results."""

        if self._disable_agent_testing:
            return

        value_table = self._valuetable._V
        state_count = self._valuetable._state_count

        episodes = 100000

        stats = play_qagent.play(
            episodes,
            False,
            False,
            env=self._env,
            base_env=self._base_env,
            value_table=value_table,
            state_count=state_count,
            feature_defs=feature_defs,
        )

        if self._azure_client:
            self._azure_client.save_test_stats(
                training_episodes=training_episodes,
                episodes=stats.episodes,
                total_wins=stats.total_wins,
                total_losses=stats.total_losses,
                pct_win=stats.pct_win,
            )

    def mean_squared_difference(self, o) -> int:
        """
        Calculates the mean squared difference between this QTable and the
        passed QTable.
        """

        return np.square(self._V - o).mean(axis=None)
