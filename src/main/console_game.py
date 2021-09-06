import CEO.cards.game as g
from CEO.cards.game import *
from CEO.cards.player import *
from CEO.cards.simplebehavior import *


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

    def lead(self, card: CardValue, count: int, index: int, player: Player):

        self._played = [None] * len(self._players)
        self._played[index] = card
        self._cur_trick_size = count

        plural = count > 1

        print(
            player.name
            + " ("
            + str(index)
            + ") leads "
            + str(count)
            + " "
            + card.to_display(plural)
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

    def __init__(self, listener: ConsoleListener):
        self._listener = listener

    def _print_hand(self, hand: Hand):
        print("Hand: ", end="")
        for i in range(13):
            count = hand.count(CardValue(i))

            for j in range(count):
                print(str(i), " ", end="")

        print("")

    def _get_card_to_play(self, hand: Hand):
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

    def pass_cards(self, hand: Hand, count: int) -> list[CardValue]:

        print("Pass", count, "cards")
        self._print_hand(hand)

        ret = []
        while len(ret) < count:
            ret.append(self._get_card_to_play(hand))

        return ret

    def lead(self, hand: Hand, state: RoundState) -> CardValue:

        self._listener.print_cur_players(state)

        print("Lead:")
        self._print_hand(hand)

        return self._get_card_to_play(hand)

    def play_on_trick(
        self, hand: Hand, cur_trick_value: CardValue, cur_trick_count: int, state: RoundState
    ) -> CardValue:

        self._listener.print_cur_trick(state)

        print("Current trick:", str(cur_trick_count), "cards of", str(cur_trick_value))
        self._print_hand(hand)

        while True:
            ret = self._get_card_to_play(hand)

            if ret == None:
                return ret
            elif ret.value <= cur_trick_value.value:
                print("Invalid lead", ret)
                continue

            return ret


def main():
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


if __name__ == "__main__":
    main()
