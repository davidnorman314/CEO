"""Program that trains an agent based on a JSON configuration file.
   Example configuration files are in the data directory in the root of
   the repository
   """

import argparse
import json
import random

from learning.qlearning_traces import QLearningTraces
from azure_rl.azure_client import AzureClient

import cProfile
from pstats import SortKey

from gym_ceo.envs.seat_ceo_env import SeatCEOEnv, ActionEnum, CEOActionSpace
from gym_ceo.envs.seat_ceo_features_env import SeatCEOFeaturesEnv
from CEO.cards.eventlistener import EventListenerInterface, PrintAllEventListener

def main():
    """Main function"""
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
        "--seed",
        type=int,
        nargs=1,
        default=None,
        help="The random seed",
    )
    parser.add_argument(
        "--pickle-file",
        type=str,
        nargs="?",
        default=None,
        help="The name of the file where pickled results should be saved.",
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
        "--azure",
        dest="azure",
        action="store_const",
        const=True,
        default=False,
        help="Save agent and log information to azure blob storage.",
    )
    parser.add_argument('configfile', metavar='config_file', type=str, nargs=1,
                        help='The learning configuration file')

    args = parser.parse_args()

    print("Loading configuration from", args.configfile[0])
    with open(args.configfile[0]) as f:
        config = json.load(f)

    # Create the arguments for the learning object
    kwargs = dict()
    if args.azure:
        kwargs["azure_client"] = AzureClient()

    kwargs["train_episodes"] = config["episodes"]

    do_log = False
    if args.log:
        do_log = args.log

    if args.seed:
        print("Arg seed", args.seed)
        random.seed(args.seed[0])
    else:
        print("No value")
        random.seed(0)

    listener = PrintAllEventListener()
    listener = EventListenerInterface()
    base_env = SeatCEOEnv(listener=listener)
    env = SeatCEOFeaturesEnv(base_env)

    learning_type = config["learning_type"]
    if learning_type == "qlearning_traces":
        learning = QLearningTraces(env, **kwargs)
    else:
        print("Unknown learning type", learning_type)
        exit(1)

    if args.profile:
        print("Running with profiling")
        cProfile.run("qlearning.train()", sort=SortKey.CUMULATIVE)
    else:
        learning.train(do_log)

    # Save the agent in a pickle file.
    if args.pickle_file:
        print("Saving results to", args.pickle_file)
        learning.pickle("qlearning_traces", args.pickle_file)
    else:
        print("Not saving results to local file.")


if __name__ == "__main__":
    main()
