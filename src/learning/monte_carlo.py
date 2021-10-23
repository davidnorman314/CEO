"""Monte carlo reinforcement learning for CEO.

The program can either train a new model or play games with a trained model.
"""

import gym
import random
import pickle
import argparse
from typing import List, Tuple
from copy import copy, deepcopy
import numpy as np
from learning.learning_base import LearningBase
from collections import deque

from gym_ceo.envs.seat_ceo_env import SeatCEOEnv
from gym_ceo.envs.seat_ceo_features_env import SeatCEOFeaturesEnv
from CEO.cards.eventlistener import EventListenerInterface, GameWatchListener, PrintAllEventListener
from CEO.cards.deck import Deck
from CEO.cards.hand import Hand, CardValue


class MonteCarloLearning(LearningBase):
    greedy_count: int
    search_count: int

    _base_env: gym.Env

    def __init__(self, env: gym.Env, base_env: gym.Env):
        super().__init__(env)

        self._base_env = base_env

        self.greedy_count = 0
        self.search_count = 0

    def set_base_env(self, base_env: gym.Env):
        self._base_env = base_env

    def _pick_action(self, state_tuple: tuple, do_greedy: bool):
        # The number of times we have visited this state
        n_state = np.sum(self._state_count[(*state_tuple, slice(None))])
        max_value = np.max(self._Q[(*state_tuple, slice(None))])
        min_value = np.min(self._Q[(*state_tuple, slice(None))])

        # Decide if we will be greedy
        epsilon = n_state / (100 + n_state)
        rand = random.uniform(0, 1)

        if not do_greedy:
            # Pick the greedy choice if the random number is large and the q values are different.
            do_greedy = rand >= epsilon and max_value != min_value

        # print(n_state, epsilon, rand, do_greedy)

        # Pick the action
        if do_greedy:
            self.greedy_count += 1
            action = np.argmax(self._Q[(*state_tuple, slice(None))])

            # Clip the action, if necessary. This biases the exploration
            # toward leading the lowest card.
            if action >= self._env.action_space.n:
                action = self._env.action_space.n - 1
        else:
            self.search_count += 1
            action = self._env.action_space.sample()

        return action

    def do_episode(
        self, do_greedy: bool = False, hands: list[Hand] = None, log_state: bool = False
    ) -> Tuple[List[tuple], List[int], float]:
        """Plays a hand. Returns a list of states visited, actions taken, and the reward"""
        # Reseting the environment each time as per requirement
        state = self._env.reset(hands)
        state_tuple = tuple(state.astype(int))

        # Starting the tracker for the rewards
        episode_reward = 0
        episode_explore_count = 0
        episode_exploit_count = 0

        episode_states = []
        episode_actions = []

        # Run until the episode is finished
        while True:
            action = self._pick_action(state_tuple, do_greedy)

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

            state_action_tuple = state_tuple + (action,)

            # Perform the action
            new_state, reward, done, info = self._env.step(action)

            episode_states.append(state_tuple)
            episode_actions.append(action)

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

        return episode_states, episode_actions, episode_reward

    def train(self, episodes: int):
        # prev_qtable = deepcopy(self._Q)
        prev_qtable = None

        total_training_reward = 0
        recent_episode_rewards = deque()
        max_recent_episode_rewards = 10000
        states_visited = 0

        for episode in range(episodes):
            states, actions, reward = self.do_episode()

            # Update q
            for i in range(len(actions)):
                state = states[i]
                action = actions[i]

                state_action_tuple = state + (action,)

                if self._state_count[state_action_tuple] == 0:
                    states_visited += 1

                self._state_count[state_action_tuple] += 1
                alpha = 1 / self._state_count[state_action_tuple]
                self._Q[state_action_tuple] += alpha * (reward - self._Q[state_action_tuple])

            # Update the search status
            total_training_reward += reward
            recent_episode_rewards.append(reward)
            if len(recent_episode_rewards) > max_recent_episode_rewards:
                recent_episode_rewards.popleft()

            last_search = 0
            last_greedy = 0
            if episode > 0 and episode % 2000 == 0:
                ave_training_rewards = total_training_reward / (episode + 1)
                recent_rewards = sum(recent_episode_rewards) / len(recent_episode_rewards)
                recent_search_count = self.search_count - last_search
                recent_greedy_count = self.greedy_count - last_greedy
                search_ratio = recent_search_count / (recent_search_count + recent_greedy_count)

                print(
                    "Episode {} Ave rewards {:.3f} Recent rewards {:.3f} States visited {} Search ratio {:.3f}".format(
                        episode,
                        ave_training_rewards,
                        recent_rewards,
                        states_visited,
                        search_ratio,
                    )
                )

                last_search = self.search_count
                last_greedy = self.greedy_count

            if prev_qtable is not None and episode > 0 and episode % 2000 == 0:
                err = self.mean_squared_difference(prev_qtable)
                prev_qtable = deepcopy(self._Q)

                print("Iteration ", episode, "delta", err)


def train_and_save(episodes: int):
    # Set up the environment
    random.seed(0)
    listener = PrintAllEventListener()
    listener = EventListenerInterface()
    base_env = SeatCEOEnv(listener=listener)
    env = SeatCEOFeaturesEnv(base_env)

    learning = MonteCarloLearning(env, base_env)
    learning.train(episodes)

    # Save the agent in a pickle file.
    learning.pickle("monte_carlo", "monte_carlo.pickle")


def play(episodes: int):
    # Load the agent
    with open("monte_carlo.pickle", "rb") as f:
        learning = pickle.load(f)

    # Set up the environment
    random.seed(0)

    listener = PrintAllEventListener()
    listener = GameWatchListener("RL")
    base_env = SeatCEOEnv(listener=listener)
    env = SeatCEOFeaturesEnv(base_env)
    learning.set_env(env)
    learning.set_base_env(base_env)

    # Play the episodes
    for count in range(episodes):
        print("Playing episode", count + 1)
        deck = Deck(base_env.num_players)
        hands = deck.deal()
        save_hands = deepcopy(hands)

        states, actions, reward = learning.do_episode(True, hands, True)

        if reward < 0:
            file = "play_hands/hands" + str(count + 1) + ".pickle"
            with open(file, "wb") as f:
                pickle.dump(save_hands, f, pickle.HIGHEST_PROTOCOL)

            for i in range(len(states)):
                print(i, "state", states[i], "action", actions[i])


def play_round(round_pickle_file: str):
    # Load the agent
    with open("monte_carlo.pickle", "rb") as f:
        learning = pickle.load(f)

    # Load the hands
    with open(round_pickle_file, "rb") as f:
        hands = pickle.load(f)

    # Set up the environment
    random.seed(0)

    listener = PrintAllEventListener()
    listener = GameWatchListener("RL")
    base_env = SeatCEOEnv(listener=listener)
    env = SeatCEOFeaturesEnv(base_env)
    learning.set_env(env)

    states, actions, reward = learning.do_episode(True, hands, True)

    for i in range(len(states)):
        print(i, "state", states[i], "action", actions[i])

        for a in range(base_env.max_action_value):
            print(
                "  action",
                a,
                "value",
                learning._Q[(*states[i], a)],
                "count",
                learning._state_count[(*states[i], a)],
            )


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
        "--play",
        dest="play",
        action="store_const",
        const=True,
        help="Have a trained agent play games",
    )

    parser.add_argument(
        "--play-round",
        dest="play_round_file",
        type=str,
        default=None,
        help="The name of a pickle file containing a list of hands",
    )

    args = parser.parse_args()
    # print(args)

    if args.train:
        train_and_save(args.episodes)
    elif args.play_round_file:
        play_round(args.play_round_file)
    elif args.play:
        play(args.episodes)
    else:
        parser.print_usage()


if __name__ == "__main__":
    # execute only if run as a script
    main()