from CEO.cards.hand import *
from CEO.cards.player import *


class EventListenerInterface:
    """
    Interface implemented by objects that receive events in the game.
    """

    def start_round(self, players: list[Player]):
        pass

    def pass_cards(
        self,
        cards: list[CardValue],
        from_index: int,
        from_player: Player,
        to_index: int,
        to_player: Player,
    ):
        pass

    def before_lead(self, index: int, player: Player, hand: Hand, state: RoundState):
        pass

    def lead(self, cards: CardValue, count: int, index: int, player: Player):
        pass

    def before_play_cards(
        self,
        starting_position: int,
        player_position: int,
        player: Player,
        hand: Hand,
        cur_trick_value: CardValue,
        cur_trick_count: int,
        state: RoundState,
    ):
        pass

    def play_cards(self, cards: CardValue, count: int, index: int, player: Player):
        pass

    def pass_on_trick(self, index: int, player: Player):
        pass

    def end_round(self, next_round_players: list[Player]):
        pass


class PrintAllEventListener(EventListenerInterface):
    """
    Interface that prints all events. This includes information that would
    normally be secret.
    """

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

    def before_lead(self, index: int, player: Player, hand: Hand, state: RoundState):
        print(player.name, "leads on a trick. Hand", hand)

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

    def before_play_cards(
        self,
        starting_position: int,
        player_position: int,
        player: Player,
        hand: Hand,
        cur_trick_value: CardValue,
        cur_trick_count: int,
        state: RoundState,
    ):
        print(player.name, "plays on a trick. Hand", hand)

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


class GameWatchListener(EventListenerInterface):
    """
    Interface that prints information about the game to the console in a format
    useful to seen how a plater is watching the game.
    """

    _player_name: str

    _players: list[Player]
    _played: list[CardValue]
    _cur_trick_size: int
    _start_seat: int

    def __init__(self, player_name: str):
        self._player_name = player_name

    def start_round(self, players: list[Player]):
        player_name_list = [player.name for player in players]
        names_str = " ".join(player_name_list)

        print("Starting round:", names_str)

        self._players = players

        for i in range(len(players)):
            if players[i].name == self._player_name:
                self._start_seat = i
                break

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

    def before_lead(self, index: int, player: Player, hand: Hand, state: RoundState):
        if player.name != self._player_name:
            return

        print("Before leading:")
        self.print_cur_players(state)
        self.print_hand(hand)

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

    def before_play_cards(
        self,
        starting_position: int,
        player_position: int,
        player: Player,
        hand: Hand,
        cur_trick_value: CardValue,
        cur_trick_count: int,
        state: RoundState,
    ):
        if player.name != self._player_name:
            return

        print("Before playing on trick:")
        self.print_cur_trick(state)
        print("Current trick:", str(cur_trick_count), "cards of", str(cur_trick_value))
        self.print_hand(hand)

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
        width = 13

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

    def print_hand(self, hand: Hand):
        print("Hand: ", end="")
        for i in range(13):
            count = hand.count(CardValue(i))

            for j in range(count):
                print(str(i), " ", end="")

        print("")

    def end_round(self, next_round_players: list[Player]):
        for i in range(len(next_round_players)):
            if next_round_players[i].name == self._player_name:
                print("Previous seat ", self._start_seat)
                print("    Next seat ", i)
