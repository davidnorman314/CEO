import CEO.cards.game as g
from CEO.cards.game import *
from CEO.cards.player import *
from CEO.cards.simplebehavior import *


class ConsoleBehavior(PlayerBehaviorInterface):
    """
    Class that gets the cards to play from the console
    """

    def _print_hand(self, hand: Hand):
        print("Hand: ", end="")
        for i in range(13):
            count = hand.count(CardValue(i))

            for j in range(count):
                print(str(i), " ", end="")

        print("")

    def _get_card_value(self):
        valStr = input()

        if valStr == "pass":
            return None

        ret = CardValue(int(valStr))

        # print("Playing", ret)

        return ret

    def pass_cards(self, hand: Hand, count: int) -> list[CardValue]:

        print("Pass", count, "cards")
        self._print_hand(hand)

        ret = []
        while len(ret) < count:
            ret.append(self._get_card_value())

        return ret

    def lead(self, hand: Hand, state: RoundState) -> CardValue:

        print("Lead:")
        self._print_hand(hand)

        return self._get_card_value()

    def play_on_trick(
        self, hand: Hand, cur_trick_value: CardValue, cur_trick_count: int, state: RoundState
    ) -> CardValue:

        print("Current trick:", str(cur_trick_count), "cards of", str(cur_trick_value))
        self._print_hand(hand)

        return self._get_card_value()


class ConsoleListener(EventListenerInterface):
    """
    Interface that prints information about the game to the console.
    """

    _player_name: str

    def __init__(self, player_name: str):
        self._player_name = player_name

    def start_round(self, players: list[Player]):
        player_name_list = [player.name for player in players]
        names_str = " ".join(player_name_list)

        print("Starting round:", names_str)

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
        print(player.name + " (" + str(index) + ") passes")


def main():
    print("Staring game.")

    human_name = "David"
    console_listener = ConsoleListener(human_name)

    human = Player(human_name, ConsoleBehavior())

    players = [human]
    for i in range(5):
        name = "Basic" + str(i + 1)
        players.append(Player(name, BasicBehavior()))

    game = g.Game(players, console_listener)
    game.play(3)


if __name__ == "__main__":
    main()
