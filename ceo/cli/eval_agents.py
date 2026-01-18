"""CLI to play rounds with various agents and evaluate their performance."""

import argparse

from ceo.learning.eval_agents import evaluate_agents, print_statistics


def main():
    parser = argparse.ArgumentParser(description="Play many games")
    parser.add_argument(
        "--print",
        dest="print",
        action="store_const",
        const=True,
        default=False,
        help="Print the game status.",
    )
    parser.add_argument(
        "--num-rounds",
        dest="num_rounds",
        type=int,
        default=1000,
        help="The number of rounds to play",
    )
    parser.add_argument(
        "--num-players",
        dest="num_players",
        type=int,
        default=None,
        help="The number of players in the game.",
    )
    parser.add_argument(
        "--ppo-agents",
        dest="ppo_agents",
        type=str,
        nargs="*",
        default=[],
        help=(
            "Specifies directories containing trained PPO agents "
            "to include in the games."
        ),
    )
    parser.add_argument(
        "--basic-agent-seats",
        dest="basic_agent_seats",
        type=int,
        nargs="*",
        default=[],
        help="Specifies which seats should be played by BasicBehavior agents.",
    )
    parser.add_argument(
        "--device",
        dest="device",
        type=str,
        default=None,
        help="The CUDA device to use, e.g., cuda or cuda:0",
    )

    args = parser.parse_args()

    result = evaluate_agents(
        num_players=args.num_players,
        num_rounds=args.num_rounds,
        ppo_agents=args.ppo_agents,
        basic_agent_seats=args.basic_agent_seats,
        device=args.device,
        do_print=args.print,
    )

    if result is None:
        # Print mode was used
        exit(1)

    win_loss_listener, heuristic_monitor, players = result
    print_statistics(
        win_loss_listener,
        heuristic_monitor,
        players,
        args.num_players,
        args.num_rounds,
    )


if __name__ == "__main__":
    main()
