"""CLI to play rounds using an agent based on a Q table."""

import argparse
import pickle
import random

from ceo.learning.play_qagent import play, play_round


def main():
    parser = argparse.ArgumentParser(description="Play rounds using a trained agent.")
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
        dest="episodes",
        type=int,
        default=100000,
        help="The number of rounds to play",
    )

    parser.add_argument(
        "--agent-file",
        dest="agent_file",
        type=str,
        help="The pickle file containing the agent",
    )

    parser.add_argument(
        "--ppo-dir",
        dest="ppo_dir",
        type=str,
        help=(
            "The zip file containing the PPO agent. "
            "The directory should have best_model.zip and params.json."
        ),
    )

    parser.add_argument(
        "--azure-agent",
        dest="azure_agent",
        type=str,
        help="The name of the Auzre blob containing the pickled agent.",
    )

    parser.add_argument(
        "--play",
        dest="play",
        action="store_const",
        const=True,
        help="Have a trained agent play games",
    )

    parser.add_argument(
        "--play-round-file",
        dest="play_round_file",
        type=str,
        default=None,
        help="The name of a pickle file containing a list of hands",
    )

    parser.add_argument(
        "--device",
        dest="device",
        type=str,
        default=None,
        help="The name of the CUDA device to use.",
    )

    parser.add_argument(
        "--do-logging",
        dest="do_logging",
        action="store_const",
        const=True,
        default=False,
        help="Log information giving the details of each hand.",
    )

    parser.add_argument(
        "--save-failed-hands",
        dest="save_failed_hands",
        action="store_const",
        const=True,
        default=False,
        help="Save pickle files for hands where the agent got a negative reward.",
    )

    parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=None,
        help="Set the random seed.",
    )

    args = parser.parse_args()

    agent_args = dict()
    if args.agent_file:
        agent_args["local_file"] = args.agent_file
    elif args.ppo_dir:
        agent_args["ppo_dir"] = args.ppo_dir
    elif args.azure_agent:
        agent_args["azure_blob_name"] = args.azure_agent
    else:
        print("No agent file specified.")
        return

    if args.device:
        agent_args["device"] = args.device

    if args.seed is not None:
        random.seed(args.seed)
    else:
        random.seed()

    if args.play_round_file:
        # Load the hands
        with open(args.play_round_file, "rb") as f:
            hands = pickle.load(f)

        play_round(hands, args.do_logging, **agent_args)
    elif args.play:
        _stats = play(
            args.episodes, args.do_logging, args.save_failed_hands, **agent_args
        )
    else:
        parser.print_usage()


if __name__ == "__main__":
    main()
