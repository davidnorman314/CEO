# from collections.abc
# import CEO.CEO.cards.hand as hand
# import CEO.cards.hand as hand
# import .hand as hand
# import hand as hand
# from hand import *
import random as random

from ceo.game.hand import CardValue, Hand


class Deck:
    """
    Class representing a deck of cards.
    """

    def __init__(self, suit_count: int):
        assert suit_count >= 1

        self._suit_count = suit_count

    def deal(self) -> list[Hand]:
        hands = []

        # Create the deck
        cards = []
        for _ in range(self._suit_count):
            cards.extend([CardValue(index) for index in range(13)])

        # Shuffle
        random.shuffle(cards)

        # Create the hands
        next = 0
        for _ in range(self._suit_count):
            hand = Hand()

            for _ in range(13):
                hand.add_cards(cards[next], 1)
                next += 1

            hands.append(hand)

        return hands
