"""Monte carlo reinforcement learning for CEO.

The program can either train a new model or play games with a trained model.
"""

import argparse
import random
from collections import deque
from copy import deepcopy
from multiprocessing import Pool, RawArray

import gymnasium

from ceo.envs.actions import ActionEnum
from ceo.envs.seat_ceo_env import CEOActionSpace, SeatCEOEnv
from ceo.envs.seat_ceo_features_env import SeatCEOFeaturesEnv
from ceo.game.eventlistener import (
    EventListenerInterface,
    PrintAllEventListener,
)
from ceo.learning.learning_base import QTableLearningBase


class SearchStatistics:
    greedy_count: int
    explore_count: int

    def __init__(self):
        self.greedy_count = 0
        self.explore_count = 0

    def add(self, other):
        self.greedy_count += other.greedy_count
        self.explore_count += other.explore_count


class MonteCarloLearning(QTableLearningBase):
    _base_env: gymnasium.Env

    def __init__(self, env: gymnasium.Env, base_env: gymnasium.Env, **kwargs):
        """Constructor for a learning object.
        The kwargs are passed to the QTable constructor so it can be initialized
        for multiprocessing.
        """
        super().__init__("monte_carlo", env, **kwargs)

        self._base_env = base_env

    def set_base_env(self, base_env: gymnasium.Env):
        self._base_env = base_env

    def _pick_action(
        self,
        state_tuple: tuple,
        action_space: CEOActionSpace,
        statistics: SearchStatistics,
    ) -> ActionEnum:
        # If the action space only has one action, return it
        if action_space.n == 1:
            return action_space.actions[0]

        # The number of times we have visited this state
        n_state = self._qtable.visit_count(state_tuple, action_space)
        min_value, max_value = self._qtable.min_max_value(state_tuple, action_space)

        # Decide if we will be greedy
        epsilon = n_state / (100 + n_state)
        rand = random.uniform(0, 1)

        # Pick the greedy choice if the random number is small and the q values
        # are different.
        do_greedy = rand <= epsilon and max_value != min_value

        # print(n_state, epsilon, rand, do_greedy)

        # Pick the action
        if do_greedy:
            statistics.greedy_count += 1
            action = self._qtable.greedy_action(state_tuple, action_space)
            # action = np.argmax(self._Q[(*state_tuple, slice(None))])
        else:
            statistics.explore_count += 1
            action_space = self._env.action_space
            action_index = action_space.sample()
            action = action_space.actions[action_index]

        assert isinstance(action, ActionEnum)

        return action

    def do_episode(
        self, log_state: bool = False
    ) -> tuple[list[tuple], list[int], float, SearchStatistics]:
        """Plays a hand. Returns a list of states visited, actions taken,
        and the reward"""

        statistics = SearchStatistics()

        # Reseting the environment each time as per requirement
        state = self._env.reset()
        state_tuple = tuple(state.astype(int))

        # Starting the tracker for the rewards
        episode_reward = 0
        _episode_explore_count = 0
        _episode_exploit_count = 0

        episode_states = []
        episode_actions = []

        # Run until the episode is finished
        while True:
            action = self._pick_action(state_tuple, self._env.action_space, statistics)

            if log_state:
                print("State", state_tuple)
                print("Action", action)

                for a in range(self._base_env.max_action_value):
                    print(
                        "  action",
                        a,
                        "value",
                        self._Q[(*state_tuple, a)],
                        "count",
                        self._state_count[(*state_tuple, a)],
                    )

            _state_action_tuple = state_tuple + (action,)

            # Perform the action
            action_index = self._env.action_space.actions.index(action)
            new_state, reward, done, info = self._env.step(action_index)

            episode_states.append(state_tuple)
            episode_actions.append(action)

            # print("state", state)
            # print("state_tuple", state_tuple)
            # print("action", action)
            # print("done", done)

            if new_state is not None:
                new_state_tuple = tuple(new_state.astype(int))
                _new_state_value = self._qtable.state_value(
                    new_state_tuple, self._env.action_space
                )
                # new_state_value = np.max(self._Q[(*new_state_tuple, slice(None))])
            else:
                assert done
                assert reward != 0

                new_state_tuple = None
                _new_state_value = 0

            # print("hand 2", info["hand"])
            # print("New q", type(self._Q[state_action_tuple]))
            # print("Q shape", self._Q.shape)
            # print("State len", len(state))
            # print("State shape", state.shape)

            # Increasing our total reward and updating the state
            episode_reward += reward
            state = new_state
            state_tuple = new_state_tuple

            # See if the episode is finished
            if done:
                # print("Reward", reward)
                break

        return episode_states, episode_actions, episode_reward, statistics

    def train(self, episodes: int, process_count: int):
        # prev_qtable = deepcopy(self._Q)
        prev_qtable = None

        total_training_reward = 0
        recent_episode_rewards = deque()
        max_recent_episode_rewards = 10000
        states_visited = 0
        episodes_per_work_task = 50
        episode_count = 0

        statistics = SearchStatistics()

        last_status_log = 0
        last_explore = 0
        last_greedy = 0

        pool = None
        if process_count > 1:
            q_raw_array, state_count_raw_array = self._qtable.get_shared_arrays()

            pool = Pool(
                processes=process_count,
                initializer=init_worker,
                initargs=(q_raw_array, state_count_raw_array),
            )

        while episode_count < episodes:
            episode_results = []

            if process_count == 1:
                # Do a single episode in process
                states, actions, reward, statistics = self.do_episode()
                episode_results.append((states, actions, reward, statistics))
            else:
                # Do many episodes in child processes
                results = pool.map(
                    worker_func, [episodes_per_work_task] * process_count
                )
                for result in results:
                    episode_results.extend(result)

            # Update q for each episode
            for states, actions, reward, this_statistics in episode_results:
                episode_count += 1
                statistics.add(this_statistics)

                for i in range(len(actions)):
                    state = states[i]
                    action = actions[i]

                    state_action_tuple = state + (action,)

                    state_action_count = self._qtable.state_visit_count(
                        state_action_tuple
                    )
                    if state_action_count == 0:
                        states_visited += 1

                    self._qtable.increment_state_visit_count(state_action_tuple)
                    state_action_count += 1

                    state_action_value = self._qtable.state_action_value(
                        state_action_tuple
                    )
                    alpha = 1.0 / state_action_count
                    self._qtable.update_state_visit_value(
                        state_action_tuple, alpha * (reward - state_action_value)
                    )

                # Update the search status
                total_training_reward += reward
                recent_episode_rewards.append(reward)
                if len(recent_episode_rewards) > max_recent_episode_rewards:
                    recent_episode_rewards.popleft()

            # Log information about the search
            if episode_count - last_status_log >= 2000:
                last_status_log = episode_count
                ave_training_rewards = total_training_reward / (episode_count + 1)
                recent_rewards = sum(recent_episode_rewards) / len(
                    recent_episode_rewards
                )
                recent_explore_count = statistics.explore_count - last_explore
                recent_greedy_count = statistics.greedy_count - last_greedy
                explore_fraction = 0
                if recent_explore_count + recent_greedy_count > 0:
                    explore_fraction = recent_explore_count / (
                        recent_explore_count + recent_greedy_count
                    )

                print(
                    f"Episode {episode_count} Ave rewards {ave_training_rewards:.3f} "
                    f"Recent rewards {recent_rewards:.3f} "
                    f"States visited {states_visited} "
                    f"Explore fraction {explore_fraction:.3f}"
                )

                last_explore = statistics.explore_count
                last_greedy = statistics.greedy_count

            if (
                prev_qtable is not None
                and episode_count > 0
                and episode_count % 5000 == 0
            ):
                err = self.mean_squared_difference(prev_qtable)
                prev_qtable = deepcopy(self._Q)

                print("Iteration ", episode_count, "delta", err)

        print("Finished with", episode_count, "episodes")

        if pool is not None:
            pool.close()


def create_environment(**kwargs):
    """Initialize the environment and learning objects.
    If kwargs is empty, do normal, single-process learning.
    If kwargs has parent=True, then create a parent environment for learning
    using sub-processes to run the episodes.
    If kwargs has shared_q=RawArray and shared_state_count=RawArray, then create
    the worker environment using the given shared arrays.
    """

    env_kwargs = dict()
    if not kwargs:
        # Single process
        pass
    elif "parent" in kwargs and kwargs["parent"]:
        env_kwargs["shared"] = True
    elif "shared_q" in kwargs and "shared_state_count" in kwargs:
        env_kwargs["shared_q"] = kwargs["shared_q"]
        env_kwargs["shared_state_count"] = kwargs["shared_state_count"]
    else:
        raise Exception("Illegal arguments to create_environment")

    listener = PrintAllEventListener()
    listener = EventListenerInterface()
    base_env = SeatCEOEnv(listener=listener)
    env = SeatCEOFeaturesEnv(base_env)

    learning = MonteCarloLearning(env, base_env, **env_kwargs)

    return base_env, env, learning


def train_and_save(episodes: int, process_count: int):
    # Set up the environment
    random.seed(0)

    env_kwargs = dict()
    if process_count > 1:
        env_kwargs["parent"] = True
    base_env, env, learning = create_environment(**env_kwargs)

    learning.train(episodes, process_count)

    # Save the agent in a pickle file.
    learning.pickle("monte_carlo.pickle")


worker_base_env = None
worker_env = None
worker_learning = None


def init_worker(q_raw_array: RawArray, state_count_raw_array: RawArray):
    global worker_base_env
    global worker_env
    global worker_learning

    worker_base_env, worker_env, worker_learning = create_environment(
        shared_q=q_raw_array, shared_state_count=state_count_raw_array
    )


def worker_func(episode_count: int):
    assert worker_learning is not None
    episode_results = []

    for _episode in range(episode_count):
        states, actions, reward, statistics = worker_learning.do_episode()
        episode_results.append((states, actions, reward, statistics))

    return episode_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        dest="profile",
        action="store_const",
        const=True,
        default=False,
        help="Do profiling.",
    )

    parser.add_argument(
        "--train",
        dest="train",
        action="store_const",
        const=True,
        help="Train a new agent",
    )
    parser.add_argument(
        "--episodes",
        dest="episodes",
        type=int,
        default=100000,
        help="The number of rounds to play",
    )
    parser.add_argument(
        "--processes",
        dest="process_count",
        type=int,
        default=1,
        help="The number of parallel processes to use during training",
    )

    args = parser.parse_args()
    # print(args)

    if args.train:
        train_and_save(args.episodes, args.process_count)
    else:
        parser.print_usage()


if __name__ == "__main__":
    # execute only if run as a script
    main()
