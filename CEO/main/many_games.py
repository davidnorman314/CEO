import json
import CEO.cards.game as g
from CEO.cards.game import *
from CEO.cards.player import *
from CEO.cards.simplebehavior import *


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

    listener = PrintAllEventListener()
    listener = EventListenerInterface()
    game = g.Game(players, listener)
    game.play(round_count=1000, do_shuffle=False)


if __name__ == "__main__":
    main()
