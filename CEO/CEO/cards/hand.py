import array

class CardValue:
    def __init__(self, value:int):
        assert value >= 0
        assert value <= 12

        self.value = value

    def __eq__(self, other):
        if isinstance(other, CardValue):
            return self.value == other.value
        return NotImplemented

class PlayedCards:
    def __init__(self, value : CardValue, count : int):
        assert count >= 1

        self.value = value
        self.count = count

class Hand:
    """
    Class representing a CEO hand
    """

    def __init__(self):
        self._cards = array.array('i', [0,0,0,0,0,0,0,0,0,0,0,0,0] )

    def add_cards(self, card_value : CardValue, count :int):
        """
        Adds cards with the given value to the hand
        """
        assert isinstance(card_value, CardValue)

        self._cards[card_value.value] += count

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