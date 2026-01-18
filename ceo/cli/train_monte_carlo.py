"""CLI for Monte Carlo reinforcement learning training."""

import argparse
import random

from ceo.envs.seat_ceo_env import SeatCEOEnv
from ceo.envs.seat_ceo_features_env import SeatCEOFeaturesEnv
from ceo.game.eventlistener import EventListenerInterface, PrintAllEventListener
from ceo.learning.monte_carlo import MonteCarloLearning


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


def main():
    parser = argparse.ArgumentParser(
        description="Monte carlo reinforcement learning for CEO."
    )
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

    if args.train:
        train_and_save(args.episodes, args.process_count)
    else:
        parser.print_usage()


if __name__ == "__main__":
    main()
