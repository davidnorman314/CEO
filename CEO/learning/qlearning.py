import gym
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import copy
from collections import deque

import cProfile
from pstats import SortKey

from gym_ceo.envs.seat_ceo_env import SeatCEOEnv
from gym_ceo.envs.seat_ceo_features_env import SeatCEOFeaturesEnv
from CEO.cards.eventlistener import EventListenerInterface, PrintAllEventListener


class QLearning:
    """
    Class implementing q-learning for an OpenAI gym
    """

    _env: gym.Env
    _Q: np.ndarray
    _action_index: int
    _train_episodes: int
    _max_action_value: int

    def __init__(self, env: gym.Env, train_episodes=100000):
        self._env = env
        self._train_episodes = train_episodes
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
        self._action_index = len(obs_space.low)

        self._Q = np.zeros(q_dims, dtype=np.float32)
        print("Q dims", q_dims)
        print("Q table size", self._Q.nbytes // (1024 * 1024), "mb")

    def train(self):
        # Creating lists to keep track of reward and epsilon values
        alpha = 0.2  # learning rate
        discount_factor = 0.7
        epsilon = 1
        max_epsilon = 0.5
        min_epsilon = 0.01
        decay = 0.00001

        test_episodes = 100
        max_steps = 100

        print("Starting training with", self._train_episodes, "episodes")

        # Training the agent
        total_training_reward = 0
        recent_episode_rewards = deque()
        recent_explore_counts = deque()
        recent_exploit_counts = deque()
        max_recent_episode_rewards = 10000
        states_visited = 0
        for episode in range(1, self._train_episodes + 1):
            # Reseting the environment each time as per requirement
            state = self._env.reset()
            state_tuple = tuple(state.astype(int))

            # Starting the tracker for the rewards
            episode_reward = 0
            episode_explore_count = 0
            episode_exploit_count = 0

            # Per-episode logging
            episode_states = []
            episode_value_before = []
            episode_value_after = []
            episode_hands = []

            # Run until the episode is finished
            while True:
                # Choose if we will explore or exploit
                exp_exp_sample = random.uniform(0, 1)

                exploit_action = np.argmax(self._Q[(*state_tuple, slice(None))])

                # Clip the action, if necessary. This biases the exploration
                # toward leading the lowest card.
                if exploit_action >= env.action_space.n:
                    exploit_action = env.action_space.n - 1

                explore_action = self._env.action_space.sample()

                if exp_exp_sample > epsilon or exploit_action == explore_action:
                    episode_exploit_count += 1

                    action = exploit_action

                    # print("q action", action, type(action))
                    # print("  expected reward", self._Q[state, action])
                else:
                    action = explore_action

                    episode_explore_count += 1
                    # print("e action", action, type(action))

                state_action_tuple = state_tuple + (action,)

                # Perform the action
                new_state, reward, done, info = self._env.step(action)

                # print("state", state)
                # print("state_tuple", state_tuple)
                # print("action", action)
                # print("done", done)

                if new_state is not None:
                    new_state_tuple = tuple(new_state.astype(int))
                    new_state_value = np.max(self._Q[(*new_state_tuple, slice(None))])
                else:
                    assert done
                    assert reward != 0

                    new_state_tuple = None
                    new_state_value = 0

                # Update the Q-table using the Bellman equation
                episode_states.append(state_action_tuple)
                episode_value_before.append(self._Q[state_action_tuple])
                if self._Q[state_action_tuple] == 0:
                    states_visited += 1
                self._Q[state_action_tuple] = self._Q[state_action_tuple] + alpha * (
                    reward + discount_factor * new_state_value - self._Q[state_action_tuple]
                )
                episode_value_after.append(self._Q[state_action_tuple])
                episode_hands.append(copy.deepcopy(info))

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
                if done == True:
                    # print("Reward", reward)
                    break

            # Cutting down on exploration by reducing the epsilon
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)

            # Save the reward
            total_training_reward += episode_reward
            recent_episode_rewards.append(episode_reward)
            recent_exploit_counts.append(episode_exploit_count)
            recent_explore_counts.append(episode_explore_count)
            if len(recent_episode_rewards) > max_recent_episode_rewards:
                recent_episode_rewards.popleft()
                recent_explore_counts.popleft()
                recent_exploit_counts.popleft()

            if episode > 0 and episode % 2000 == 0:
                ave_training_rewards = total_training_reward / episode
                recent_rewards = sum(recent_episode_rewards) / len(recent_episode_rewards)
                recent_explore_rate = sum(recent_explore_counts) / (
                    sum(recent_exploit_counts) + sum(recent_explore_counts)
                )

                print(
                    "Episode {} Ave rewards {:.3f} Recent rewards {:.3f} Explore rate {:.3f} States visited {}".format(
                        episode,
                        ave_training_rewards,
                        recent_rewards,
                        recent_explore_rate,
                        states_visited,
                    )
                )

            if episode > 0 and episode % 20000 == 0:
                # Log the states for this episode
                print("Episode info")
                for i in range(len(episode_states)):
                    print(
                        i,
                        episode_states[i],
                        episode_value_before[i],
                        episode_value_after[i],
                        episode_hands[i],
                    )
                print("Reward", episode_reward)

            if episode > 0 and episode % 100000 == 0:
                # Iterate over the entire Q array and count the number of each type of element.
                # This is very slow.
                zero_count = 0
                pos_count = 0
                neg_count = 0
                for val in np.nditer(self._Q):
                    if val == 0:
                        zero_count += 1
                    elif val > 0:
                        pos_count += 1
                    else:
                        neg_count += 1

                print("  pos", pos_count, "neg", neg_count, "zero", zero_count)


# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Do learning")
    parser.add_argument(
        "--profile",
        dest="profile",
        action="store_const",
        const=True,
        default=False,
        help="Do profiling.",
    )
    parser.add_argument(
        "--episodes",
        dest="train_episodes",
        type=int,
        default=1000,
        help="The number of rounds to play",
    )

    args = parser.parse_args()
    print(args)

    kwargs = dict()
    if args.train_episodes:
        kwargs["train_episodes"] = args.train_episodes

    random.seed(0)
    listener = PrintAllEventListener()
    listener = EventListenerInterface()
    base_env = SeatCEOEnv(listener=listener)
    env = SeatCEOFeaturesEnv(base_env)

    qlearning = QLearning(env, **kwargs)

    if args.profile:
        print("Running with profiling")
        cProfile.run("qlearning.train()", sort=SortKey.CUMULATIVE)
    else:
        qlearning.train()
