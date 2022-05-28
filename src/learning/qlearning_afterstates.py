import gym
import numpy as np
import argparse
import random
import math
from azure_rl.azure_client import AzureClient
from learning.learning_base import ValueTableLearningBase, EpisodeInfo
from collections import deque

from pstats import SortKey
from enum import Enum

from gym_ceo.envs.seat_ceo_env import SeatCEOEnv
from gym_ceo.envs.features import FeatureObservationFactory

from CEO.cards.eventlistener import EventListenerInterface, PrintAllEventListener

import cProfile


class QLearningAfterstates(ValueTableLearningBase):
    """
    Class implementing q-learning using afterstates for an OpenAI gym
    """

    feature_defs: list
    _obs_factory: FeatureObservationFactory

    class AlphaType(Enum):
        STATE_VISIT_COUNT = 1
        EPISODE_COUNT = 2
        CONSTANT = 3

    _train_episodes: int

    def __init__(self, env: gym.Env, train_episodes=100000, feature_defs=None, **kwargs):
        super().__init__("qlearning_afterstates", env, **kwargs)
        self._train_episodes = train_episodes

        if feature_defs is not None:
            self.feature_defs = feature_defs
        else:
            self.feature_defs = self.get_default_features(env)

        self._obs_factory = FeatureObservationFactory(env, self.feature_defs)

        super()._set_observation_space(self._obs_factory.observation_space)

    def train(self, params: dict, do_log: bool):
        # Validate the parameters
        discount_factor = params["discount_factor"]
        if discount_factor is None:
            raise ValueError("The parameter discount_factor is missing")

        epsilon = params["epsilon"]
        if epsilon is None:
            raise ValueError("The parameter epsilon is missing")

        max_epsilon = params["max_epsilon"]
        if max_epsilon is None:
            raise ValueError("The parameter max_epsilon is missing")

        min_epsilon = params["min_epsilon"]
        if min_epsilon is None:
            raise ValueError("The parameter min_epsilon is missing")

        decay = params["decay"]
        if decay is None:
            raise ValueError("The parameter decay is missing")

        alpha_type_str = params["alpha_type"]
        if alpha_type_str is None:
            raise ValueError("The parameter alpha_type_str is missing")

        if alpha_type_str == "state_visit_count":
            alpha_type = self.AlphaType.STATE_VISIT_COUNT
        elif alpha_type_str == "constant":
            alpha_type = self.AlphaType.CONSTANT
        elif alpha_type_str == "episode_count":
            alpha_type = self.AlphaType.EPISODE_COUNT
        else:
            raise ValueError("Invalid alpha_type: " + alpha_type_str)

        alpha_exponent = None
        alpha_constant = None
        if (
            alpha_type == self.AlphaType.STATE_VISIT_COUNT
            or alpha_type == self.AlphaType.EPISODE_COUNT
        ):
            alpha_exponent = params["alpha_exponent"]
            if alpha_exponent is None:
                return "The parameter alpha_exponent is missing"
        elif alpha_type == self.AlphaType.CONSTANT:
            alpha_constant = params["alpha"]
            if alpha_constant is None:
                return "The parameter alpha is missing"

        print("Training with", self._train_episodes, "episodes")

        # Log the start of training to Azure, if necessary.
        if self._azure_client:
            params = dict()
            params["discount_factor"] = discount_factor
            params["epsilon"] = epsilon
            params["max_epsilon"] = max_epsilon
            params["min_epsilon"] = min_epsilon
            params["decay"] = decay
            params["alpha_type"] = alpha_type_str
            if alpha_exponent is not None:
                params["alpha_exponent"] = alpha_exponent
            if alpha_constant is not None:
                params["alpha_constant"] = alpha_constant

            self._azure_client.start_training(
                "qlearning_afterstates",
                self._env.action_space_type,
                self._env.num_players,
                self._env.seat_number,
                params,
                self.feature_defs,
            )

        # Training the agent
        total_training_reward = 0
        recent_episode_rewards = deque()
        recent_explore_counts = deque()
        recent_exploit_counts = deque()
        max_recent_episode_rewards = 10000
        states_visited = 0
        for episode in range(1, self._train_episodes + 1):
            # Reseting the environment to start the new episode.
            state = self._env.reset()

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

                if exp_exp_sample > epsilon:
                    # Exploit
                    action, expected_value, new_state_visit_count = self.find_greedy_action(state)
                    action_type = "-------"

                    episode_exploit_count += 1
                else:
                    # Explore
                    action = self._env.action_space.sample()
                    action_type = "explore"

                    episode_explore_count += 1

                afterstate = self.get_afterstate(state, action)
                afterstate_tuple = tuple(afterstate.astype(int))

                # Perform the action
                new_state, reward, done, info = self._env.step(action)

                if new_state is not None:
                    # The estimated state value of new_state is the maximum expected
                    # value across all actions
                    max_action, new_state_value, new_state_visit_count = self.find_greedy_action(
                        new_state
                    )
                else:
                    assert done
                    assert reward != 0

                    new_state_value = 0

                # Update the Q-table using the Bellman equation
                episode_info = EpisodeInfo()
                episode_infos.append(episode_info)

                episode_info.state = state
                episode_info.value_before = self._valuetable.state_value(afterstate_tuple)

                self._valuetable.increment_state_visit_count(afterstate_tuple)
                state_visit_count = self._valuetable.state_visit_count(afterstate_tuple)
                if state_visit_count == 1:
                    states_visited += 1

                if alpha_type == self.AlphaType.STATE_VISIT_COUNT:
                    # Calculate the learning rate based on the state count.
                    # See Learning Rates for Q-learning, Even-Dar and Mansour, 2003
                    # https://www.jmlr.org/papers/volume5/evendar03a/evendar03a.pdf
                    # They recommend 0.85.
                    alpha = 1.0 / (state_visit_count ** alpha_exponent)
                elif alpha_type == self.AlphaType.EPISODE_COUNT:
                    # Calculate the learning rate based on the state count.
                    # See Learning Rates for Q-learning, Even-Dar and Mansour, 2003
                    # https://www.jmlr.org/papers/volume5/evendar03a/evendar03a.pdf
                    # They recommend 0.85.
                    alpha = 1.0 / (episode ** alpha_exponent)
                elif alpha_type == self.AlphaType.CONSTANT:
                    alpha = alpha_constant
                else:
                    raise ValueError("alpha_type is not defined")

                state_value = self._valuetable.state_value(afterstate_tuple)
                delta = alpha * (reward + discount_factor * new_state_value - state_value)
                self._valuetable.update_state_value(afterstate_tuple, delta)

                episode_info.value_after = self._valuetable.state_value(afterstate_tuple)
                episode_info.action_type = action_type
                episode_info.alpha = alpha
                episode_info.state_visit_count = state_visit_count

                # Increasing our total reward and updating the state
                episode_reward += reward
                state = new_state

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
                    "qlearning_afterstates",
                    episode,
                    ave_training_rewards,
                    recent_rewards,
                    recent_explore_rate,
                    states_visited,
                )

            if (
                episode > 0
                and episode % self._during_training_stats_frequency == 0
                and episode < self._train_episodes
            ):
                self.do_play_test(episode, self.feature_defs)

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

        if self._azure_client:
            self._azure_client.end_training()

        return self._search_statistics[-1]

    def get_afterstate(self, state: np.ndarray, action: int):
        """Returns the feature afterstate from the given action."""
        # Calculate the afterstate if this action is taken.
        afterstate_full_observation, played_card = self._env.get_afterstate(state, action)

        # Calcualate features for the afterstate
        info = dict()
        afterstate_feature_observation = self._obs_factory.make_feature_observation(
            afterstate_full_observation, info
        )

        return afterstate_feature_observation

    def find_greedy_action(self, state: np.ndarray) -> tuple[int, float]:
        """Checks all possible actions to find the greedy one."""

        return self._valuetable.find_greedy_action(self._env, self._obs_factory, state)

    def get_default_features(self, env: SeatCEOEnv):
        print("Using default parameters.")
        self.feature_defs = []
        half_players = env.num_players // 2
        for i in range(half_players - 1):
            feature_params = dict()
            feature_params["other_player_index"] = i
            feature_params["max_value"] = 4
            self.feature_defs.append(("OtherPlayerHandCount", feature_params))

        feature_params = dict()
        self.feature_defs.append(("WillWinTrick_AfterState", feature_params))

        min_card_exact_feature = 9
        for i in range(min_card_exact_feature, 13):
            feature_params = dict()
            feature_params["card_value_index"] = i
            self.feature_defs.append(("HandCardCount", feature_params))

        feature_params = dict()
        feature_params["threshold"] = min_card_exact_feature
        self.feature_defs.append(("SinglesUnderValueCount", feature_params))

        feature_params = dict()
        feature_params["threshold"] = min_card_exact_feature
        self.feature_defs.append(("DoublesUnderValueCount", feature_params))

        feature_params = dict()
        feature_params["threshold"] = min_card_exact_feature
        self.feature_defs.append(("TriplesUnderValueCount", feature_params))

        feature_params = dict()
        self.feature_defs.append(("TrickPosition", feature_params))

        if False:
            # This isn't useful as-is for afterstates, since the feature information
            # is relative to cards in the hand. One possible replacement feature
            # would be one that told whether later players could play on the trick, i.e.,
            # if the number of cards in their hand is larger or smaller than the trick count.
            feature_params = dict()
            self.feature_defs.append(("CurTrickValue", feature_params))

            # Also probably not useful, since it just gives a litte information about whether
            # later players will be able to play on the trick. Instead of using it, let's try
            # to add the information to WillWinTrick_AfterState
            feature_params = dict()
            self.feature_defs.append(("CurTrickCount", feature_params))

        return self.feature_defs


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

    kwargs["disable_agent_testing"] = True

    do_log = False
    if args.log:
        do_log = args.log

    random.seed(0)
    listener = PrintAllEventListener()
    listener = EventListenerInterface()
    env = SeatCEOEnv(listener=listener)

    # Set up default parameters
    params = dict()
    params["discount_factor"] = 0.7
    params["epsilon"] = 1
    params["max_epsilon"] = 0.5
    params["min_epsilon"] = 0.01
    params["decay"] = 0.0000001
    params["alpha_type"] = "state_visit_count"
    params["alpha_exponent"] = 0.60

    qlearning = QLearningAfterstates(env, **kwargs)

    if args.profile:
        print("Running with profiling")
        cProfile.run("qlearning.train(params, do_log)", sort=SortKey.CUMULATIVE)
    else:
        qlearning.train(params, do_log)

    # Save the agent in a pickle file.
    qlearning.pickle("qlearning_afterstates.pickle", feature_defs=qlearning.feature_defs)
