import argparse
import pickle
import json
from tkinter import W
from copy import deepcopy
from pathlib import Path

import CEO.cards.game as g
from CEO.cards.game import *
from CEO.cards.player import *
from CEO.cards.simplebehavior import *
from CEO.cards.round import Round, RoundState

from learning.play_qagent import play_round, create_agent


class ConsoleListener(EventListenerInterface):
    """
    Interface that prints information about the game to the console.
    """

    _player_name: str

    _players: list[Player]
    _played: list[CardValue]
    _cur_trick_size: int

    def __init__(self, player_name: str):
        self._player_name = player_name

    def start_round(self, players: list[Player]):
        player_name_list = [player.name for player in players]
        names_str = " ".join(player_name_list)

        print("Starting round:", names_str)

        self._players = players

    def pass_cards(
        self,
        cards: list[CardValue],
        from_index: int,
        from_player: Player,
        to_index: int,
        to_player: Player,
    ):

        if from_player.name != self._player_name or to_player.name != self._player_name:
            # The information is for another player's pass, so we can't see it.
            return

        card_str_list = [card.to_display() for card in cards]
        cards_str = " ".join(card_str_list)

        print(
            from_player.name
            + " ("
            + str(from_index)
            + ") passes "
            + cards_str
            + " to "
            + to_player.name
            + " ("
            + str(to_index)
            + ")"
        )

    def lead(
        self,
        cur_card_value: CardValue,
        cur_card_count: int,
        starting_player_index: int,
        cur_player: Player,
    ):

        self._played = [None] * len(self._players)
        self._played[starting_player_index] = cur_card_value
        self._cur_trick_size = cur_card_count

        plural = cur_card_count > 1

        print(
            cur_player.name
            + " ("
            + str(starting_player_index)
            + ") leads "
            + str(cur_card_count)
            + " "
            + cur_card_value.to_display(plural)
        )

    def play_cards(self, card: CardValue, count: int, index: int, player: Player):

        self._played[index] = card

        plural = count > 1

        print(
            player.name
            + " ("
            + str(index)
            + ") plays "
            + str(count)
            + " "
            + card.to_display(plural)
        )

    def pass_on_trick(self, index: int, player: Player):
        self._played[index] = "pass"

        print(player.name + " (" + str(index) + ") passes")

    def print_cur_players(self, state: RoundState):
        width = 10

        for player in self._players:
            print(player.name.ljust(width), end="")

        print("")

        for cards_remaining in state.cards_remaining:
            if cards_remaining > 0:
                text = "- " + str(cards_remaining)
            else:
                text = "Out"

            print(text.ljust(width), end="")

        print("")

        return width

    def print_cur_trick(self, state: RoundState):
        width = self.print_cur_players(state)

        for played in self._played:
            if played == None:
                print("".ljust(width), end="")
            elif played is str:
                print(played.ljust(width), end="")
            else:
                text = str(self._cur_trick_size) + " " + str(played)
                print(text.ljust(width), end="")

        print("")


class ConsoleBehavior(PlayerBehaviorInterface):
    """
    Class that gets the cards to play from the console
    """

    _listener: ConsoleListener

    is_reinforcement_learning: bool

    def __init__(self, listener: ConsoleListener):
        self._listener = listener
        self.is_reinforcement_learning = False

    def _print_hand(self, hand: Hand):
        print("Hand: ", end="")
        for i in range(13):
            count = hand.count(CardValue(i))

            for j in range(count):
                print(str(i), " ", end="")

            if count > 0:
                print(" ", end="")

        print("")

    def _get_card_to_pass(self, hand: Hand):
        while True:
            valStr = input()

            if valStr == "pass":
                return None

            try:
                int_val = int(valStr)
                ret = CardValue(int_val)
            except ValueError:
                print("Not an integer or invalid integer:", valStr)
                continue

            if hand.count(ret) == 0:
                print("You don't have any of", ret, " in your hand")
                continue

            return ret

    def _get_card_to_play(self, hand: Hand, *, lead: bool, trick_value=None, trick_count=None):
        while True:
            valStr = input()

            if valStr == "pass":
                if lead:
                    print("You must lead")
                    continue

                return None

            try:
                int_val = int(valStr)
                ret = CardValue(int_val)
            except ValueError:
                print("Not an integer or invalid integer:", valStr)
                continue

            if hand.count(ret) == 0:
                print("You don't have any of", ret, "in your hand")
                continue

            if not lead:
                hand_card_count = hand.count(ret)

                if hand_card_count < trick_count:
                    print("You don't enough", ret, "in your hand")
                    continue

                if ret.value <= trick_value.value:
                    print("The card", ret, "is too small")
                    continue

            return ret

    def pass_cards(self, hand: Hand, count: int) -> list[CardValue]:

        pass_automatically = True

        if pass_automatically:
            print("Passing automatically")
            simple_behavior = SimpleBehaviorBase()
            return simple_behavior.pass_singles(hand, count)
        else:
            print("Pass", count, "cards")
            self._print_hand(hand)

            ret = []
            while len(ret) < count:
                ret.append(self._get_card_to_pass(hand))

        return ret

    def lead(self, starting_player_index: int, hand: Hand, state: RoundState) -> CardValue:

        self._listener.print_cur_players(state)

        print("Lead:")
        self._print_hand(hand)

        return self._get_card_to_play(hand, lead=True)

    def play_on_trick(
        self,
        starting_player_index: int,
        current_player_index: int,
        hand: Hand,
        cur_trick_value: CardValue,
        cur_trick_count: int,
        state: RoundState,
    ) -> CardValue:

        self._listener.print_cur_trick(state)

        print("Current trick:", str(cur_trick_count), "cards of", str(cur_trick_value))
        self._print_hand(hand)

        while True:
            ret = self._get_card_to_play(
                hand, lead=False, trick_value=cur_trick_value, trick_count=cur_trick_count
            )

            if ret == None:
                return ret
            elif ret.value <= cur_trick_value.value:
                print("Invalid lead", ret)
                continue

            return ret


def play_game():
    print("Staring game.")

    human_name = "David"
    console_listener = ConsoleListener(human_name)

    human = Player(human_name, ConsoleBehavior(console_listener))

    players = []
    for i in range(5):
        name = "Basic" + str(i + 1)
        players.append(Player(name, BasicBehavior()))

    # Put the human player last
    players.append(human)

    game = g.Game(players, console_listener)
    game.play(round_count=500, do_shuffle=False)


def play_ceo_rounds(agent_args: dict):
    print("Playing rounds as CEO.")

    history = GameHistory()

    human_name = "David"
    console_listener = ConsoleListener(human_name)

    human = Player(human_name, ConsoleBehavior(console_listener))

    players = []

    # Put the human player first
    players.append(human)

    for i in range(5):
        name = "Basic" + str(i + 2)
        players.append(Player(name, BasicBehavior()))

    num_players = len(players)
    deck = Deck(num_players)

    while True:
        console_listener.start_round(players)

        # Gym sets the random seed, so we need to re-seed or we always get the same deal.
        random.seed()

        # Deal the cards
        hands = deck.deal()
        hands_copy = deepcopy(hands)
        hands_copy2 = deepcopy(hands)

        # Pass cards
        passcards = PassCards(players, hands, console_listener)
        passcards.do_card_passing()

        round = Round(players, hands, console_listener)
        round.play()

        next_round_order = round.get_next_round_order()
        player_won_round = next_round_order[0] == 0

        # Save the result to the history file
        if player_won_round:
            history.save_win()
        else:
            history.save_loss()

        # Have the agent play
        agent_reward = play_round(hands_copy, True, **agent_args)
        agent_won_round = agent_reward > 0.0

        print(f"Won {history.total_won} of {history.total_hands} or {history.total_won/history.total_hands}")

        # Log if there was a different result
        if player_won_round != agent_won_round:
            print("Human", "won" if player_won_round else "lost")
            print("Agent", "won" if agent_won_round else "lost")

            for inc in range(0, 1000):
                suffix = (
                    "human_"
                    + ("won" if player_won_round else "lost")
                    + "_agent_"
                    + ("won" if agent_won_round else "lost")
                )

                file = "play_hands/console_hands_" + suffix + "_" + str(history.total_hands + 1 + inc) + ".pickle"

                if Path(file).exists():
                    continue

                with open(file, "wb") as f:
                    pickle.dump(hands_copy2, f, pickle.HIGHEST_PROTOCOL)

                break
        else:
            print("Same result for player and agent.")


class GameHistory:
    history_file = "../data/history/ceo_console_game_history.ndjson"

    total_hands: int
    total_won: int

    def __init__(self):
        self.total_hands = 0
        self.total_won = 0

        # Load the current history
        lines = 0
        with open(self.history_file, "r") as f:
            for line in f:
                lines += 1
                info = json.loads(line)

                if "TotalHands" in info:
                    self.total_hands += info["TotalHands"]
                    self.total_won += info["TotalWon"]
                elif "Won" in info:
                    self.total_hands += 1
                    self.total_won += 1
                elif "Loss" in info:
                    self.total_hands += 1
                    self.total_won += 0
                else:
                    raise Exception("Unknown line in history_file")

        # See if we should compress
        if lines > 10:
            print("Compressing history file")

            info = dict()
            info["TotalHands"] = self.total_hands
            info["TotalWon"] = self.total_won

            with open(self.history_file, "w") as f:
                f.write(json.dumps(info))
                f.write("\n")

    def save_win(self):
        self.total_hands += 1
        self.total_won += 1

        info = dict()
        info["Won"] = True

        self._append(info)

    def save_loss(self):
        self.total_hands += 1

        info = dict()
        info["Loss"] = True

        self._append(info)

    def _append(self, info):
        with open(self.history_file, "a") as f:
            f.write(json.dumps(info))
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Do learning")
    parser.add_argument(
        "--play-ceo",
        dest="play_ceo",
        action="store_const",
        const=True,
        default=False,
        help="Play many rounds as CEO.",
    )
    parser.add_argument(
        "--play-game",
        dest="play_game",
        action="store_const",
        const=True,
        default=False,
        help="Play a game.",
    )
    parser.add_argument(
        "--agent-file",
        dest="agent_file",
        type=str,
        help="The pickle file containing the agent",
    )
    parser.add_argument(
        "--azure-agent",
        dest="azure_agent",
        type=str,
        help="The name of the Auzre blob containing the pickled agent.",
    )

    args = parser.parse_args()

    agent_args = dict()
    if args.agent_file:
        agent_args["local_file"] = args.agent_file
    elif args.azure_agent:
        agent_args["azure_blob_name"] = args.azure_agent
    else:
        print("No agent file specified.")

    if args.play_ceo:
        play_ceo_rounds(agent_args)
    elif args.play_game:
        play_game()
    else:
        parser.usage()


if __name__ == "__main__":
    main()
