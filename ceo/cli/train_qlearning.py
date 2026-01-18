"""CLI for Q-learning training."""

import argparse
import cProfile
import random
from pstats import SortKey

from ceo.azure_rl.azure_client import AzureClient
from ceo.envs.seat_ceo_env import SeatCEOEnv
from ceo.envs.seat_ceo_features_env import SeatCEOFeaturesEnv
from ceo.game.eventlistener import EventListenerInterface, PrintAllEventListener
from ceo.learning.qlearning import QLearning


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

    kwargs["disable_agent_testing"] = True

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
    params["lambda"] = 1e-6
    params["epsilon"] = 1
    params["max_epsilon"] = 0.5
    params["min_epsilon"] = 0.01
    params["decay"] = 0.0000001
    params["alpha_type"] = "state_visit_count"
    params["alpha_exponent"] = 0.60

    qlearning = QLearning(env, base_env, **kwargs)

    if args.profile:
        print("Running with profiling")
        cProfile.run("qlearning.train(params, do_log)", sort=SortKey.CUMULATIVE)
    else:
        qlearning.train(params, do_log)

    # Save the agent in a pickle file.
    qlearning.pickle("qlearning.pickle")


if __name__ == "__main__":
    main()
