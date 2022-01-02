import gym
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import copy
import math
from azure_rl.azure_client import AzureClient
from learning.learning_base import LearningBase, EpisodeInfo
from collections import deque

import cProfile
from pstats import SortKey

from gym_ceo.envs.seat_ceo_env import SeatCEOEnv, ActionEnum, CEOActionSpace
from gym_ceo.envs.seat_ceo_features_env import SeatCEOFeaturesEnv
from CEO.cards.eventlistener import EventListenerInterface, PrintAllEventListener


class QLearningTraces(LearningBase):
    """
    Class implementing Watkins Q(\lambda) with eligibility traces for an OpenAI gym.
    See:
        1. Sutton and Barto 312 and the
        2. The screenshot from an earlier version of Sutton and Barto in
            https://stackoverflow.com/questions/40862578/how-to-understand-watkinss-q%CE%BB-learning-algorithm-in-suttonbartos-rl-book
        3. https://towardsdatascience.com/eligibility-traces-in-reinforcement-learning-a6b458c019d6 and
    """

    _train_episodes: int

    def __init__(self, env: gym.Env, train_episodes=100000, **kwargs):
        super().__init__(env, **kwargs)
        self._train_episodes = train_episodes

    def _pick_action(self, state_tuple: tuple, epsilon: float) -> ActionEnum:
        # If there is only one action, return it
        if self._env.action_space.n == 1:
            return self._env.action_space.actions[0], 0

        # Choose if we will explore or exploit
        exp_exp_sample = random.uniform(0, 1)

        exploit_action = self._qtable.greedy_action(state_tuple, self._env.action_space)
        # exploit_action = np.argmax(self._Q[(*state_tuple, slice(None))])

        explore_action_index = self._env.action_space.sample()
        explore_action = self._env.action_space.actions[explore_action_index]

        if exp_exp_sample > epsilon or exploit_action == explore_action:
            is_exploit = True
            action = exploit_action
        else:
            is_exploit = False
            action = explore_action

        assert isinstance(action, ActionEnum)

        return action, is_exploit

    def train(self, params: dict, do_log: bool):
        # Validate the parameters
        discount_factor = params["discount_factor"]
        if discount_factor is None:
            return "The parameter discount_factor is missing"

        lambda_val = params["lambda"]
        if lambda_val is None:
            return "The parameter lambda is missing"

        epsilon = params["epsilon"]
        if epsilon is None:
            return "The parameter epsilon is missing"

        max_epsilon = params["max_epsilon"]
        if max_epsilon is None:
            return "The parameter max_epsilon is missing"

        min_epsilon = params["min_epsilon"]
        if min_epsilon is None:
            return "The parameter min_epsilon is missing"

        decay = params["decay"]
        if decay is None:
            return "The parameter decay is missing"

        print("Training with", self._train_episodes, "episodes")

        # Log the start of training to Azure, if necessary.
        if self._azure_client:
            params = dict()
            params["discount_factor"] = discount_factor
            params["lambda_val"] = lambda_val
            params["epsilon"] = epsilon
            params["max_epsilon"] = max_epsilon
            params["min_epsilon"] = min_epsilon
            params["decay"] = decay

            self._azure_client.start_training("qlearning_traces", params)

        # Training the agent
        total_training_reward = 0
        recent_episode_rewards = deque()
        recent_explore_counts = deque()
        recent_exploit_counts = deque()
        max_recent_episode_rewards = 10000
        states_visited = 0
        for episode in range(1, self._train_episodes + 1):
            if do_log:
                print("Starting episode")

            # Reseting the environment each time as per requirement
            state = self._env.reset()
            state_tuple = tuple(state.astype(int))
            info = dict()

            # Pick the initial action
            action, is_exploit = self._pick_action(state_tuple, epsilon)

            # Initialize the eligibility traces
            eligibility_traces = dict()

            # Initialize the episode information.
            episode_reward = 0
            episode_explore_count = 0
            episode_exploit_count = 0
            episode_infos: list[EpisodeInfo] = []
            episode_value_before = dict()
            episode_value_after = dict()

            # Run until the episode is finished
            while True:
                # Save off the action information
                if is_exploit:
                    episode_exploit_count += 1
                    action_type = "-------"
                else:
                    action_type = "explore"
                    episode_explore_count += 1

                if do_log:
                    print("Top of loop.")
                    action_space = self._env.action_space

                    print("State", state_tuple)
                    print("Obs info:")
                    for key, value in info.items():
                        print(" ", key, "->", value)

                    print("Action", action)
                    print("Action choice:", action_type)
                    print("Action values")
                    for a in range(self._env.full_env.max_action_value):
                        full_action = ActionEnum(a)
                        name = full_action.name if full_action in action_space.actions else ""
                        selected = "selected" if full_action == action else ""

                        print(
                            "  action",
                            a,
                            "value",
                            self._qtable.state_action_value((*state_tuple, full_action)),
                            "count",
                            self._qtable.state_visit_count((*state_tuple, full_action)),
                            name,
                        )

                # Save state information
                state_action_tuple = state_tuple + (action,)

                episode_info = EpisodeInfo()
                episode_infos.append(episode_info)
                episode_info.state = state_action_tuple
                episode_info.value_before = self._qtable.state_action_value(state_action_tuple)
                # episode_info.value_before = self._Q[state_action_tuple]
                if state_action_tuple not in episode_value_before:
                    episode_value_before[state_action_tuple] = episode_info.value_before

                self._qtable.increment_state_visit_count(state_action_tuple)
                # self._state_count[state_action_tuple] += 1
                state_visit_count = self._qtable.state_visit_count(state_action_tuple)
                # state_visit_count = self._state_count[state_action_tuple]
                if state_visit_count == 1:
                    states_visited += 1

                # Perform the action
                action_index = self._env.action_space.actions.index(action)
                state_prime, reward, done, info = self._env.step(action_index)

                if state_prime is not None:
                    if do_log:
                        print("After action")
                        action_space = self._env.action_space

                        print("  State", state_tuple)
                        print("  Obs info:")
                        for key, value in info.items():
                            print(" ", key, "->", value)

                        print("  Action values")
                        for a in range(self._env.full_env.max_action_value):
                            full_action = ActionEnum(a)
                            name = ""

                            print(
                                "    action",
                                a,
                                "value",
                                self._qtable.state_action_value((*state_tuple, full_action)),
                                "count",
                                self._qtable.state_visit_count((*state_tuple, full_action)),
                                name,
                            )

                    state_prime_tuple = tuple(state_prime.astype(int))
                    state_prime_value = self._qtable.state_value(
                        state_prime_tuple, self._env.action_space
                    )
                    # state_prime_value = np.max(self._Q[(*state_prime_tuple, slice(None))])

                    # Pick the next action
                    action_prime, is_exploit = self._pick_action(state_prime_tuple, epsilon)

                    # Find the action with the highest estimated reward
                    action_star = self._qtable.greedy_action(
                        state_prime_tuple, self._env.action_space
                    )
                    # action_star = np.argmax(self._Q[(*state_prime_tuple, slice(None))])
                    action_star_value = self._qtable.state_action_value(
                        (*state_prime_tuple, action_star)
                    )
                    # action_star_value = self._Q[(*state_prime_tuple, action_star)]
                    if action_star_value == state_prime_value:
                        action_star = action_prime

                    # Calculate the update value
                    delta = (
                        reward
                        + discount_factor
                        * self._qtable.state_action_value((*state_prime_tuple, action_star))
                        - self._qtable.state_action_value((*state_tuple, action))
                    )

                    if do_log:
                        print("Action prime", action_prime)
                        print("  is exploit", is_exploit)
                        print("Action star", action_star)
                        print("Delta", delta)
                    # delta = (
                    #     reward
                    #     + discount_factor * self._Q[(*state_prime_tuple, action_star)]
                    #     - self._Q[(*state_tuple, action)]
                    # )
                else:
                    if do_log:
                        print("Reward", reward)
                    assert done
                    assert reward != 0

                    state_prime_tuple = None
                    state_prime_value = 0
                    action_star = None
                    action_prime = None

                    # Calculate the update value
                    delta = reward - self._qtable.state_action_value((*state_tuple, action))
                    # delta = reward - self._Q[(*state_tuple, action)]

                if do_log:
                    print("delta", delta)

                # Update the eligibility trace for this state
                if state_action_tuple not in eligibility_traces:
                    eligibility_traces[state_action_tuple] = 1
                else:
                    eligibility_traces[state_action_tuple] += 1

                # Find the step size alpha
                # See Learning Rates for Q-learning, Even-Dar and Mansour, 2003
                # https://www.jmlr.org/papers/volume5/evendar03a/evendar03a.pdf
                alpha = 1.0 / (state_visit_count ** 0.85)

                # Update Q and the traces
                for update_tuple in eligibility_traces.keys():
                    value_before = self._qtable.state_action_value(update_tuple)
                    update = alpha * delta * eligibility_traces[update_tuple]
                    self._qtable.update_state_visit_value(update_tuple, update)
                    # self._Q[update_tuple] += alpha * delta * eligibility_traces[update_tuple]
                    if action_prime == action_star:
                        eligibility_traces[update_tuple] *= discount_factor * lambda_val
                    else:
                        eligibility_traces[update_tuple] = 0

                    episode_value_after[update_tuple] = self._qtable.state_action_value(
                        update_tuple
                    )

                    if do_log:
                        print(
                            "Update ",
                            update_tuple,
                            " ",
                            value_before,
                            " -> ",
                            episode_value_after[update_tuple],
                        )
                        print("  trace ", eligibility_traces[update_tuple])
                    # episode_value_after[update_tuple] = self._Q[update_tuple]

                state = state_prime
                state_tuple = state_prime_tuple
                action = action_prime

                # Save information from after the update
                episode_info.value_after = self._qtable.state_action_value(state_action_tuple)
                # episode_info.value_after = self.self._Q[state_action_tuple]
                episode_info.hand = copy.deepcopy(info)
                episode_info.action_type = action_type
                episode_info.alpha = alpha
                episode_info.state_visit_count = state_visit_count

                # Increasing our total reward and updating the state
                episode_reward += reward

                # See if the episode is finished
                if done == True:
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

            if (episode > 0 and episode % 2000 == 0) or (episode == self._train_episodes):
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

                self.add_search_statistics(
                    "qlearning_traces",
                    episode,
                    ave_training_rewards,
                    recent_rewards,
                    recent_explore_rate,
                    states_visited,
                )

            if False and episode > 0 and episode % 5000 == 0:
                # Log the states for this episode
                print("Episode info")
                for info in episode_infos:
                    max_visit_count = max(map(lambda info: info.state_visit_count, episode_infos))
                    visit_chars = math.ceil(math.log10(max_visit_count))

                    format_str = "{action:2} value {val_before:6.3f} -> {val_after:6.3f} visit {visit_count:#w#} alpha {alpha:.3e}"
                    format_str = format_str.replace("#w#", str(visit_chars))

                    print(
                        format_str.format(
                            action=info.action_type,
                            val_before=episode_value_before[info.state],
                            val_after=episode_value_after[info.state],
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

        if self._azure_client:
            self._azure_client.end_training()

        return self._search_statistics[-1]


# Main function
def main():
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
        "--log",
        dest="log",
        action="store_const",
        const=True,
        default=False,
        help="Do logging.",
    )
    parser.add_argument(
        "--episodes",
        dest="train_episodes",
        type=int,
        default=100000,
        help="The number of rounds to play",
    )
    parser.add_argument(
        "--azure",
        dest="azure",
        action="store_const",
        const=True,
        default=False,
        help="Save agent and log information to azure blob storage.",
    )

    args = parser.parse_args()
    print(args)

    kwargs = dict()
    if args.train_episodes:
        kwargs["train_episodes"] = args.train_episodes

    if args.azure:
        kwargs["azure_client"] = AzureClient()

    do_log = False
    if args.log:
        do_log = args.log

    random.seed(0)
    listener = PrintAllEventListener()
    listener = EventListenerInterface()
    base_env = SeatCEOEnv(listener=listener)
    env = SeatCEOFeaturesEnv(base_env)

    # Set up default parameters
    params = dict()
    params["discount_factor"] = 0.7
    params["lambda"] = 0.5
    params["epsilon"] = 1
    params["max_epsilon"] = 0.5
    params["min_epsilon"] = 0.01
    params["decay"] = 0.00001

    qlearning = QLearningTraces(env, **kwargs)

    if args.profile:
        print("Running with profiling")
        cProfile.run("qlearning.train()", sort=SortKey.CUMULATIVE)
    else:
        qlearning.train(params, do_log)

    # Save the agent in a pickle file.
    file_name = "qlearning_traces.pickle"
    qlearning.pickle("qlearning_traces", file_name)


if __name__ == "__main__":
    main()
