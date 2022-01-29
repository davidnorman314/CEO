import gym
import numpy as np
import pickle
import datetime
from azure_rl.azure_client import AzureClient

from learning.qtable import QTable
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
    _qtable: QTable

    _search_statistics: list[dict]
    _start_time: datetime.datetime
    _last_backup_pickle_time: datetime.datetime
    _last_azure_log_time: datetime.datetime

    _azure_client: AzureClient

    def __init__(self, env: gym.Env, base_env: gym.Env, **kwargs):
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

        if now - self._last_backup_pickle_time > datetime.timedelta(minutes=30):
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
        pickle_dict["FeatureDefs"] = self._env.feature_defs
        pickle_dict["NumPlayers"] = self._env.num_players

        data = pickle.dumps(pickle_dict, pickle.HIGHEST_PROTOCOL)

        if filename:
            print("Saving results to", filename)
            with open(filename, "wb") as f:
                f.write(data)

        # Upload to Azure, if necessary
        if self._azure_client:
            self._azure_client.upload_pickle(data=data)

    def mean_squared_difference(self, o) -> int:
        """
        Calculates the mean squared difference between this QTable and the
        passed QTable.
        """

        return np.square(self._Q - o).mean(axis=None)

    def do_play_test(self, training_episodes: int):
        """Plays many hands with the current agent and logs the results."""

        q_table = self._qtable._Q
        state_count = self._qtable._state_count

        episodes = 10000

        stats = play_qagent.play(
            episodes,
            False,
            False,
            env=self._env,
            base_env=self._base_env,
            q_table=q_table,
            state_count=state_count,
        )

        if self._azure_client:
            self._azure_client.save_train_stats(
                training_episodes=training_episodes,
                episodes=stats.episodes,
                total_wins=stats.total_wins,
                total_losses=stats.total_losses,
                pct_win=stats.pct_win,
            )
