from CEO.cards.deck import *
from CEO.cards.hand import *
from CEO.cards.player import *
from CEO.cards.eventlistener import *


class PassCards:
    """
    Class that handles passing cards after the deal and before the first trick.
    """

    def __init__(self, players: list[Player], hands: list[Hand], listener: EventListenerInterface):
        self._listener = listener
        self._players = players
        self._hands = hands
        self._player_count = len(self._players)
        self._next_round_order = []

        assert len(self._players) == len(self._hands)

    def do_card_passing(self):
        # print("Passing cards")

        pass_count = self._player_count // 2

        # Pass from the lower half of the table to the upper
        for i in range(pass_count):
            from_index = self._player_count - 1 - i
            to_index = i
            from_hand = self._hands[from_index]
            to_hand = self._hands[to_index]

            cards_to_pass = pass_count - i

            # print("Passing cards from ", from_index, " ", self._players[from_index].name,
            #    " to ", to_index, " ", self._players[to_index].name)

            cards = []
            for j in range(cards_to_pass):
                cv = from_hand.max_card_value()
                from_hand.remove_cards(cv, 1)
                to_hand.add_cards(cv, 1)
                cards.append(cv)

            self._listener.pass_cards(
                cards, from_index, self._players[from_index], to_index, self._players[to_index]
            )

        # Pass from the upper half to the lower
        for i in range(pass_count):
            from_index = i
            to_index = self._player_count - 1 - i
            from_hand = self._hands[i]
            to_hand = self._hands[to_index]

            cards_to_pass = pass_count - i

            # print("Passing cards from ", from_index, " ", self._players[from_index].name,
            #    " to ", to_index, " ", self._players[to_index].name)

            values = self._players[from_index].behavoir.pass_cards(
                self._hands[from_index], cards_to_pass
            )

            assert len(values) == cards_to_pass

            for cv in values:
                from_hand.remove_cards(cv, 1)
                to_hand.add_cards(cv, 1)

            self._listener.pass_cards(
                values, from_index, self._players[from_index], to_index, self._players[to_index]
            )
