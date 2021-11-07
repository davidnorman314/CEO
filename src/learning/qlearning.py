import gym
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import copy
import math
from learning.learning_base import LearningBase, EpisodeInfo
from collections import deque

import cProfile
from pstats import SortKey

from gym_ceo.envs.seat_ceo_env import SeatCEOEnv, CEOActionSpace
from gym_ceo.envs.actions import ActionEnum
from gym_ceo.envs.seat_ceo_features_env import SeatCEOFeaturesEnv
from CEO.cards.eventlistener import EventListenerInterface, PrintAllEventListener


class QLearning(LearningBase):
    """
    Class implementing q-learning for an OpenAI gym
    """

    _train_episodes: int

    def __init__(self, env: gym.Env, train_episodes=100000):
        super().__init__(env)
        self._train_episodes = train_episodes

    def train(self):
        # Creating lists to keep track of reward and epsilon values
        discount_factor = 0.7
        epsilon = 1
        max_epsilon = 0.5
        min_epsilon = 0.01
        decay = 0.000001

        test_episodes = 100
        max_steps = 100

        print("Training with", self._train_episodes, "episodes")

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

            # Information about the episode
            episode_infos: list[EpisodeInfo] = []

            # Run until the episode is finished
            while True:
                # Choose if we will explore or exploit
                exp_exp_sample = random.uniform(0, 1)

                exploit_action = self._qtable.greedy_action(state_tuple, self._env.action_space)
                # exploit_action = np.argmax(self._Q[(*state_tuple, slice(None))])

                explore_action_index = self._env.action_space.sample()
                explore_action = self._env.action_space.actions[explore_action_index]

                if exp_exp_sample > epsilon or exploit_action == explore_action:
                    episode_exploit_count += 1

                    action = exploit_action
                    action_type = "-------"

                    # print("q action", action, type(action))
                    # print("  expected reward", self._Q[state, action])
                else:
                    action = explore_action
                    action_type = "explore"

                    episode_explore_count += 1
                    # print("e action", action, type(action))

                state_action_tuple = state_tuple + (action,)

                # Perform the action
                action_index = self._env.action_space.actions.index(action)
                new_state, reward, done, info = self._env.step(action_index)

                # print("state", state)
                # print("state_tuple", state_tuple)
                # print("action", action)
                # print("done", done)

                if new_state is not None:
                    new_state_tuple = tuple(new_state.astype(int))
                    new_state_value = self._qtable.state_value(
                        new_state_tuple, self._env.action_space
                    )
                    # new_state_value = np.max(self._Q[(*new_state_tuple, slice(None))])
                else:
                    assert done
                    assert reward != 0

                    new_state_tuple = None
                    new_state_value = 0

                # Update the Q-table using the Bellman equation
                episode_info = EpisodeInfo()
                episode_infos.append(episode_info)

                episode_info.state = state_action_tuple
                episode_info.value_before = self._qtable.state_action_value(state_action_tuple)
                # episode_info.value_before = self._Q[state_action_tuple]

                self._qtable.increment_state_visit_count(state_action_tuple)
                # self._state_count[state_action_tuple] += 1
                state_visit_count = self._qtable.state_visit_count(state_action_tuple)
                # state_visit_count = self._state_count[state_action_tuple]
                if state_visit_count == 1:
                    states_visited += 1

                # Calculate the learning rate based on the state count.
                # See Learning Rates for Q-learning, Even-Dar and Mansour, 2003
                # https://www.jmlr.org/papers/volume5/evendar03a/evendar03a.pdf
                alpha = 1.0 / (state_visit_count ** 0.85)

                state_action_value = self._qtable.state_action_value(state_action_tuple)
                delta = alpha * (reward + discount_factor * new_state_value - state_action_value)
                self._qtable.update_state_visit_value(state_action_tuple, delta)
                # self._Q[state_action_tuple] = self._Q[state_action_tuple] + alpha * (
                #    reward + discount_factor * new_state_value - self._Q[state_action_tuple]
                # )

                episode_info.value_after = self._qtable.state_action_value(state_action_tuple)
                episode_info.hand = copy.deepcopy(info)
                episode_info.action_type = action_type
                episode_info.alpha = alpha
                episode_info.state_visit_count = state_visit_count

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

            if False and episode > 0 and episode % 20000 == 0:
                # Log the states for this episode
                print("Episode info")
                for info in episode_infos:
                    max_visit_count = max(map(lambda info: info.state_visit_count, episode_infos))
                    visit_chars = math.ceil(math.log10(max_visit_count))

                    format_str = "{action:2} value {val_before:6.3f} -> {val_after:6.3f} visit {visit_count:#w#} -> {alpha:.3e} {hand}"
                    format_str = format_str.replace("#w#", str(visit_chars))

                    print(
                        format_str.format(
                            action=info.action_type,
                            val_before=info.value_before,
                            val_after=info.value_after,
                            hand=info.hand,
                            state=info.state,
                            alpha=info.alpha,
                            visit_count=info.state_visit_count,
                        )
                    )
                print("Reward", episode_reward)
                print("Epsilon {:.5f}".format(epsilon))

            if False and episode > 0 and episode % 100000 == 0:
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
        default=100000,
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

    # Save the agent in a pickle file.
    qlearning.pickle("qlearning", "qlearning.pickle")