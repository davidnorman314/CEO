"""Code to play rounds with various agents and evaluate their performance."""
import json
import argparse
from CEO.cards.game import Game
from CEO.cards.player import *
from CEO.cards.simplebehavior import *
from CEO.cards.heuristicbehavior import *
from CEO.cards.behaviorstatistics import BehaviorStatisticsCollector
from CEO.cards.eventlistener import GameWatchListener, PrintAllEventListener

from learning.ppo_agents import process_ppo_agents


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
        help="Specifies directories containing trained PPO agents to include in the games.",
    )
    parser.add_argument(
        "--device",
        dest="device",
        type=str,
        default=None,
        help="The CUDA device to use, e.g., cuda or cuda:0",
    )

    args = parser.parse_args()
    num_players = args.num_players

    # Load the agents
    custom_behaviors, custom_behavior_descs = process_ppo_agents(
        args.ppo_agents, device=args.device, num_players=num_players
    )

    players = []
    for seat in range(args.num_players):
        if seat not in custom_behaviors:
            raise Exception(f"No player specified for seat {seat}.")

        players.append(Player(custom_behavior_descs[seat], custom_behaviors[seat]))

    if args.print:
        doStats = False
        listener = PrintAllEventListener()
    else:
        doStats = True
        listener = BehaviorStatisticsCollector(players)

    num_rounds = args.num_rounds
    game = Game(players, listener)
    game.play(round_count=args.num_rounds, do_shuffle=False, reorder_seats=False)

    format1 = "{:20}"
    format2 = "{:5.2f}"
    format2a = "{:5.0f}"

    if not doStats:
        exit(1)

    # Print statistics
    for behavior_name in listener.stats:
        print(behavior_name)
        stats = listener.stats[behavior_name]

        print(format1.format("Percent in seat:"), end="")
        for i in range(num_players):
            pct = (stats.end_position_count[i] / stats.players_with_behavior_count) / num_rounds

            print(format2.format(pct), end="")

        print("")

        total_move_up = 0
        print(format1.format("Move up count:"), end="")
        for i in range(num_players):
            if i == 0:
                continue

            cnt = stats.move_up_delta[i] / stats.players_with_behavior_count
            total_move_up += cnt

            print(format2a.format(cnt), end="")

        print("")

        total_move_down = 0
        print(format1.format("Move down count:"), end="")
        for i in range(num_players):
            if i == 0:
                continue

            cnt = stats.move_down_delta[i] / stats.players_with_behavior_count
            total_move_down += cnt

            print(format2a.format(cnt), end="")

        print("")

        print(format1.format("Up count:"), end="")
        print(format2a.format(total_move_up), end="")
        print("")
        print(format1.format("Stay count:"), end="")
        print(
            format2a.format(stats.stay_count / stats.players_with_behavior_count),
            end="",
        )
        print("")
        print(format1.format("Down count:"), end="")
        print(format2a.format(total_move_down), end="")
        print("")

        print("All changes")
        for i in range(num_players):
            total_for_start_seat = 0
            for j in range(num_players):
                total_for_start_seat += stats.start_to_finish[i][j]

            print(format1.format("Start seat " + str(i)), end="")
            for j in range(num_players):
                val = stats.start_to_finish[i][j] / total_for_start_seat
                print(format2.format(val), end="")
            print("")

        # Bottom half stats
        move_up_count = stats.bottom_half_move_up / stats.players_with_behavior_count
        move_down_count = stats.bottom_half_move_down / stats.players_with_behavior_count
        stay_count = stats.bottom_half_stay / stats.players_with_behavior_count
        bottom_half_count = move_up_count + move_down_count + stay_count

        print("Bottom half stats:")
        print(format1.format("    Move up:"), end="")
        print(
            format2a.format(move_up_count),
            format2.format(move_up_count / bottom_half_count),
            end="",
        )
        print("")
        print(format1.format("       Stay:"), end="")
        print(
            format2a.format(stay_count),
            format2.format(stay_count / bottom_half_count),
            end="",
        )
        print("")
        print(format1.format("  Move down:"), end="")
        print(
            format2a.format(move_down_count),
            format2.format(move_down_count / bottom_half_count),
            end="",
        )
        print("")

        # for i in range(num_players):
        #    pct = stats.end_position_count[i]
        #
        #    print("{0:5d}".format(pct), end="")

        print("")


if __name__ == "__main__":
    main()
