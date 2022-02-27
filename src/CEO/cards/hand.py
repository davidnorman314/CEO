import array
from abc import ABC, abstractmethod


class CardValue:
    """
    Class representing a card value, e.g, 2, 3, 4, ..., King, and Ace.
    The values are zero through thirteen.
    """

    value: int

    def is_ace(self) -> bool:
        return self.value == 12

    def __init__(self, value: int):
        if value < 0:
            raise ValueError("Value is negative " + str(value))

        if value > 12:
            raise ValueError("Value is too large " + str(value))

        self.value = value

    def to_display(self, plural=True):
        return str(self)

    def __eq__(self, other):
        if isinstance(other, CardValue):
            return self.value == other.value
        return NotImplemented

    def __str__(self):
        return "V" + str(self.value)

    def __repr__(self):
        return str(self)


class PlayedCards:
    """
    Class a set of cards that are played from a hand to the table.
    """

    def __init__(self, value: CardValue, count: int):
        assert count >= 1

        self.value = value
        self.count = count


class HandInterface(ABC):
    """Abstract base class for Hand classes."""

    @abstractmethod
    def card_count(self) -> int:
        pass

    @abstractmethod
    def count(self, card_value: CardValue) -> int:
        pass

    @abstractmethod
    def max_card_value(self) -> CardValue:
        pass

    @abstractmethod
    def play_cards(self, cards: PlayedCards):
        pass


class Hand(HandInterface):
    """
    Class representing a CEO hand
    """

    _total_cards: int

    def __init__(self):
        self._cards = array.array("i", [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self._total_cards = 0

    def add_cards(self, card_value: CardValue, count: int):
        """
        Adds cards with the given value to the hand
        """
        assert card_value is not None
        assert isinstance(card_value, CardValue)

        self._cards[card_value.value] += count
        self._total_cards += count

    def remove_cards(self, card_value: CardValue, count: int):
        """
        Removes cards with the given value to the hand
        """
        assert card_value is not None
        assert isinstance(card_value, CardValue)

        assert self._cards[card_value.value] >= count

        self._cards[card_value.value] -= count
        self._total_cards -= count

    def is_empty(self) -> bool:
        return self._total_cards == 0

    def card_count(self) -> int:
        return self._total_cards

    def count(self, card_value: CardValue) -> int:
        return self._cards[card_value.value]

    def max_card_value(self) -> CardValue:
        for i in range(len(self._cards) - 1, -1, -1):
            if self._cards[i] > 0:
                return CardValue(i)

        assert False

    def play_cards(self, cards: PlayedCards):
        assert cards.count > 0

        index = cards.value.value

        assert self._cards[index] >= cards.count

        self._cards[index] -= cards.count
        self._total_cards -= cards.count

    def to_dict(self):
        """
        Converts the hand to a dictionary mapping the card value to the
        count of cards in the hand
        """
        return {key: self._cards[key] for key in range(13) if self._cards[key] > 0}

    def cards_equal(self, card_dict: dict):
        """
        Checks that hand exactly equals the hand described by the dictionary.
        The dictionary is a map from integer card value to the count of cards.
        This is intended for unit testing.
        """

        # Check that all the entries in card_dict match _cards
        for key in card_dict:
            hand_count = self._cards[key]
            dict_count = card_dict[key]
            if hand_count != dict_count:
                return False

        # Check that all the entries in _cards are also in card_dict
        # We don't need to check values here, since they were checked
        # above.
        for key in range(13):
            if self._cards[key] == 0:
                continue

            if key not in card_dict:
                return False

        return True

    def __str__(self):
        return "Hand " + str(self.to_dict())

    def __repr__(self):
        return str(self)
