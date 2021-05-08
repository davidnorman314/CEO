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

    def lead(self, cards: CardValue, count: int, index: int, player: Player):
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
