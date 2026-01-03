"""Program that trains an agent based on a JSON configuration file.
Example configuration files are in the data directory in the root of
the repository
"""

import argparse
import cProfile
import json
import random
from pstats import SortKey

import learning.play_qagent as play_qagent
from azure_rl.azure_client import AzureClient
from CEO.cards.eventlistener import EventListenerInterface, PrintAllEventListener
from gym_ceo.envs.seat_ceo_env import SeatCEOEnv
from gym_ceo.envs.seat_ceo_features_env import SeatCEOFeaturesEnv
from learning.qlearning import QLearning
from learning.qlearning_afterstates import QLearningAfterstates
from learning.qlearning_traces import QLearningTraces


def do_learning(
    configfile: str,
    do_azure: bool,
    do_logging: bool,
    random_seed: int,
    do_profile: bool,
    pickle_file: str,
    disable_agent_testing: bool,
    post_train_stats_episodes: int,
    during_training_stats_episodes: int = None,
    during_training_stats_frequency: int = None,
):
    print("Loading configuration from", configfile)
    with open(configfile) as f:
        config = json.load(f)

    # Create the arguments for the learning object
    kwargs = dict()
    azure_client = None
    if do_azure:
        print("Saving progress and results to Azure")
        azure_client = AzureClient()
        kwargs["azure_client"] = azure_client

    kwargs["train_episodes"] = config["episodes"]
    kwargs["disable_agent_testing"] = disable_agent_testing
    kwargs["during_training_stats_episodes"] = during_training_stats_episodes
    kwargs["during_training_stats_frequency"] = during_training_stats_frequency

    if random_seed is not None:
        print("Random seed", random_seed)
        random.seed(random_seed)
    else:
        random.seed(0)

    env_kwargs = dict()
    if "action_space_type" in config:
        env_kwargs["action_space_type"] = config["action_space_type"]

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
    base_env = SeatCEOEnv(listener=listener, **env_kwargs)

    learning_type = config["learning_type"]
    learning = None
    if learning_type == "qlearning_traces":
        env = SeatCEOFeaturesEnv(base_env, feature_defs=feature_defs)
        learning = QLearningTraces(env, base_env, **kwargs)
    elif learning_type == "qlearning_afterstates":
        learning = QLearningAfterstates(base_env, feature_defs=feature_defs, **kwargs)
    elif learning_type == "qlearning":
        env = SeatCEOFeaturesEnv(base_env, feature_defs=feature_defs)
        learning = QLearning(env, base_env, **kwargs)
    else:
        print("Unknown learning type", learning_type)
        exit(1)

    params = config["params"]

    final_search_statistics = None
    if do_profile:
        print("Running with profiling")
        print(learning)
        locals = {"learning": learning, "params": params, "do_logging": do_logging}
        globals = {}
        cProfile.runctx(
            "learning.train(params, do_logging)",
            locals,
            globals,
            sort=SortKey.CUMULATIVE,
        )
    else:
        final_search_statistics = learning.train(params, do_logging)

    # Save the agent in a pickle file.
    if pickle_file or azure_client:
        learning.pickle(pickle_file, feature_defs=feature_defs)

    # Run a final test of the agent, if necessary
    if post_train_stats_episodes:
        post_train_test_stats(
            learning, env, base_env, post_train_stats_episodes, azure_client
        )

    return final_search_statistics


def post_train_test_stats(learning, env, base_env, episodes, azure_client):
    q_table = learning._qtable._Q
    state_count = learning._qtable._state_count

    stats = play_qagent.play(
        episodes,
        False,
        False,
        env=env,
        base_env=base_env,
        q_table=q_table,
        state_count=state_count,
    )

    if azure_client:
        azure_client.save_post_train_test_stats(
            episodes=stats.episodes,
            total_wins=stats.total_wins,
            total_losses=stats.total_losses,
            pct_win=stats.pct_win,
        )


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
    parser.add_argument(
        "--post-train-stats-episodes",
        type=int,
        default=None,
        help="Number of episodes to run when testing after training.",
    )
    parser.add_argument(
        "--during-training-stats-episodes",
        type=int,
        default=None,
        help="Number of episodes to run for testing during training.",
    )
    parser.add_argument(
        "--during-training-stats-frequency",
        type=int,
        default=None,
        help="How frequently during training the agent should be tested.",
    )
    parser.add_argument(
        "--disable-agent-testing",
        action="store_const",
        const=True,
        default=False,
        help="Disables testing the agent during search.",
    )

    args = parser.parse_args()

    kwargs = dict()
    if args.during_training_stats_episodes:
        kwargs["during_training_stats_episodes"] = args.during_training_stats_episodes
    if args.during_training_stats_frequency:
        kwargs["during_training_stats_frequency"] = args.during_training_stats_frequency

    do_learning(
        args.configfile[0],
        args.azure,
        args.log,
        args.seed,
        args.profile,
        args.pickle_file,
        args.disable_agent_testing,
        args.post_train_stats_episodes,
        **kwargs,
    )


if __name__ == "__main__":
    main()
