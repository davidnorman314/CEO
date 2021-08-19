import gym
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random

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
        training_rewards = []
        epsilons = []

        alpha = 0.7  # learning rate
        discount_factor = 0.8
        epsilon = 1
        max_epsilon = 1
        min_epsilon = 0.01
        decay = 0.0001

        test_episodes = 100
        max_steps = 100

        print("Starting training with", self._train_episodes, "episodes")

        # Training the agent
        for episode in range(1, self._train_episodes + 1):
            print("episode", episode)
            # Reseting the environment each time as per requirement
            state = self._env.reset()
            state = state.astype(int)

            # Starting the tracker for the rewards
            total_training_rewards = 0

            # Run until the episode is finished
            while True:
                # Choose if we will explore or exploit
                exp_exp_sample = random.uniform(0, 1)

                if exp_exp_sample > epsilon:
                    action = np.argmax(self._Q[(*state, slice(None))])
                    # print("q action", action, type(action))
                    # print("  expected reward", self._Q[state, action])
                else:
                    action = self._env.action_space.sample()
                    # print("e action", action, type(action))

                # Perform the action
                new_state, reward, done, info = self._env.step(action)

                if new_state is None:
                    assert done

                # Fix the array types
                state_tuple = tuple(state.astype(int))
                if new_state is not None:
                    new_state_tuple = tuple(new_state.astype(int))

                # Update the Q-table using the Bellman equation
                # print("state", state)
                # print("state_tuple", state_tuple)
                # print("action", action)
                # print("done", done)

                if new_state is not None:
                    new_state_value = np.max(self._Q[(*new_state_tuple, slice(None))])
                else:
                    new_state_value = 0

                state_action_tuple = state_tuple + (action,)

                self._Q[state_action_tuple] = self._Q[state_action_tuple] + alpha * (
                    reward + discount_factor * new_state_value - self._Q[state_action_tuple]
                )
                # print("New q", type(self._Q[state_action_tuple]))
                # print("Q shape", self._Q.shape)
                # print("State len", len(state))
                # print("State shape", state.shape)

                # Increasing our total reward and updating the state
                total_training_rewards += reward
                state = new_state

                # See if the episode is finished
                if done == True:
                    # print("Reward", reward)
                    break

            # Cutting down on exploration by reducing the epsilon
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)

            # Update the total reward
            training_rewards.append(total_training_rewards)

            # Save epsilon
            epsilons.append(epsilon)

            if episode > 0 and episode % 2 == 0:
                total_reward = sum(training_rewards)
                ave_training_rewards = total_reward / episode

                print(
                    "Episode {} Ave rewards {} Total rewards {}".format(
                        episode, ave_training_rewards, total_reward
                    )
                )


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
    listener = EventListenerInterface()
    base_env = SeatCEOEnv(listener=listener)
    env = SeatCEOFeaturesEnv(base_env)

    qlearning = QLearning(env, **kwargs)

    if args.profile:
        print("Running with profiling")
        cProfile.run("qlearning.train()", sort=SortKey.CUMULATIVE)
    else:
        qlearning.train()
