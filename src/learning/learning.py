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


def do_learning(
    configfile: str,
    do_azure: bool,
    do_logging: bool,
    random_seed: int,
    do_profile: bool,
    pickle_file: str,
):
    print("Loading configuration from", configfile)
    with open(configfile) as f:
        config = json.load(f)

    # Create the arguments for the learning object
    kwargs = dict()
    if do_azure:
        print("Saving progress and results to Azure")
        kwargs["azure_client"] = AzureClient()

    kwargs["train_episodes"] = config["episodes"]

    if random_seed is not None:
        print("Random seed", random_seed)
        random.seed(random_seed)
    else:
        random.seed(0)

    # Get the feature definitions, if any.
    feature_defs = None
    if "features" in config:
        feature_defs = []

        for feature_config in config["features"]:
            type = feature_config["type"]
            feature_params = feature_config["params"]

            feature_defs.append((type, feature_params))

    # Create the environment
    listener = PrintAllEventListener()
    listener = EventListenerInterface()
    base_env = SeatCEOEnv(listener=listener)
    env = SeatCEOFeaturesEnv(base_env, feature_defs=feature_defs)

    learning_type = config["learning_type"]
    if learning_type == "qlearning_traces":
        learning = QLearningTraces(env, **kwargs)
    else:
        print("Unknown learning type", learning_type)
        exit(1)

    params = config["params"]

    final_search_statistics = None
    if do_profile:
        print("Running with profiling")
        cProfile.run("qlearning.train()", sort=SortKey.CUMULATIVE)
    else:
        final_search_statistics = learning.train(params, do_logging)

    # Save the agent in a pickle file.
    if pickle_file:
        print("Saving results to", pickle_file)
        learning.pickle("qlearning_traces", pickle_file)

    return final_search_statistics


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
    parser.add_argument(
        "configfile",
        metavar="config_file",
        type=str,
        nargs=1,
        help="The learning configuration file",
    )

    args = parser.parse_args()

    do_learning(
        args.configfile[0],
        args.azure,
        args.log,
        args.seed,
        args.profile,
        args.pickle_file,
    )


if __name__ == "__main__":
    main()
