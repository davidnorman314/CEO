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

    round_count = 1000
    game = g.Game(players, listener)
    game.play(round_count=round_count, do_shuffle=False)

    # Print statistics
    for behavior_name in listener.stats:
        print(behavior_name)
        stats = listener.stats[behavior_name]

        for i in range(player_count):
            pct = (stats.end_position_count[i] / stats.players_with_behavior_count) / round_count

            print("{0:5.2f}".format(pct), end="")

        print("")

        for i in range(player_count):
            pct = stats.end_position_count[i]

            print("{0:5d}".format(pct), end="")

        print("")
        print("")


if __name__ == "__main__":
    main()
