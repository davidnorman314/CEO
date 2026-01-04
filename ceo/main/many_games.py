import argparse
import json

import ceo.game.game as g
from ceo.game.eventlistener import GameWatchListener, PrintAllEventListener
from ceo.game.game import Player
from ceo.game.heuristicbehavior import HeuristicBehavior  # noqa: F401
from ceo.game.simplebehavior import BasicBehavior  # noqa: F401
from ceo.game.winlossstatistics import WinLossStatisticsCollector


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
        "--count",
        dest="count",
        type=int,
        default=1000,
        help="The number of rounds to play",
    )

    args = parser.parse_args()
    print(args)

    print("Loading from main/players.json.")
    with open("main/players.json") as f:
        data = json.load(f)

    players = []
    players_data = data["players"]
    for data in players_data:
        name = data["name"]
        behavior_class_name = data["behavior"]
        behavior_class = globals()[behavior_class_name]
        behavior = behavior_class()
        players.append(Player(name, behavior))

        if data["behavior"] == "HeuristicBehavior":
            console_log_player = data["name"]

    player_count = len(players_data)

    if args.print:
        do_stats = False
        listener = PrintAllEventListener()
        listener = GameWatchListener(console_log_player)
        print("Logging information for", console_log_player)
    else:
        do_stats = True
        listener = WinLossStatisticsCollector(players)

    round_count = args.count
    game = g.Game(players, listener)
    game.play(round_count=round_count, do_shuffle=False)

    format1 = "{:20}"
    format2 = "{:5.2f}"
    format2a = "{:5.0f}"

    if not do_stats:
        exit(1)

    # Print statistics
    for behavior_name in listener.stats:
        print(behavior_name)
        stats = listener.stats[behavior_name]

        print(format1.format("Percent in seat:"), end="")
        for i in range(player_count):
            pct = (
                stats.end_position_count[i] / stats.players_with_behavior_count
            ) / round_count

            print(format2.format(pct), end="")

        print("")

        total_move_up = 0
        print(format1.format("Move up count:"), end="")
        for i in range(player_count):
            if i == 0:
                continue

            cnt = stats.move_up_delta[i] / stats.players_with_behavior_count
            total_move_up += cnt

            print(format2a.format(cnt), end="")

        print("")

        total_move_down = 0
        print(format1.format("Move down count:"), end="")
        for i in range(player_count):
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
        for i in range(player_count):
            total_for_start_seat = 0
            for j in range(player_count):
                total_for_start_seat += stats.start_to_finish[i][j]

            print(format1.format("Start seat " + str(i)), end="")
            for j in range(player_count):
                val = stats.start_to_finish[i][j] / total_for_start_seat
                print(format2.format(val), end="")
            print("")

        # Bottom half stats
        move_up_count = stats.bottom_half_move_up / stats.players_with_behavior_count
        move_down_count = (
            stats.bottom_half_move_down / stats.players_with_behavior_count
        )
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

        # for i in range(player_count):
        #    pct = stats.end_position_count[i]
        #
        #    print("{0:5d}".format(pct), end="")

        print("")


if __name__ == "__main__":
    main()
