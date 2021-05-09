import json
import CEO.cards.game as g
from CEO.cards.game import *
from CEO.cards.player import *
from CEO.cards.simplebehavior import *
from CEO.cards.heuristicbehavior import *
from CEO.cards.behaviorstatistics import *


def main():
    print("Loading from main/players.json.")
    with open("main/players.json") as f:
        data = json.load(f)

    players = []
    playersData = data["players"]
    for data in playersData:
        name = data["name"]
        behaviorClassName = data["behavior"]
        behaviorClass = globals()[behaviorClassName]
        behavior = behaviorClass()
        players.append(Player(name, behavior))

    player_count = len(playersData)

    listener = PrintAllEventListener()
    listener = BehaviorStatisticsCollector(players)

    round_count = 10000
    game = g.Game(players, listener)
    game.play(round_count=round_count, do_shuffle=False)

    format1 = "{:20}"
    format2 = "{:5.2f}"
    format2a = "{:5.0f}"

    # Print statistics
    for behavior_name in listener.stats:
        print(behavior_name)
        stats = listener.stats[behavior_name]

        print(format1.format("Percent in seat:"), end="")
        for i in range(player_count):
            pct = (stats.end_position_count[i] / stats.players_with_behavior_count) / round_count

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
        print(format2a.format(stats.stay_count / stats.players_with_behavior_count), end="")
        print("")
        print(format1.format("Down count:"), end="")
        print(format2a.format(total_move_down), end="")
        print("")

        # for i in range(player_count):
        #    pct = stats.end_position_count[i]
        #
        #    print("{0:5d}".format(pct), end="")

        print("")


if __name__ == "__main__":
    main()
