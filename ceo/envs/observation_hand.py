from ceo.envs.observation import Observation
from ceo.game.hand import CardValue, HandInterface, PlayedCards


class ObservationHand(HandInterface):
    """Class representing a hand described by an Observation."""

    _observation: Observation
    _hand_begin_index: int

    def __init__(self, observation: Observation):
        self._observation = observation.copy()
        self._hand_begin_index = observation._factory._obs_index_hand_cards

    def card_count(self) -> int:
        array = self._observation.get_array()
        return sum(array[self._hand_begin_index : self._hand_begin_index + 13])

    def count(self, card_value: CardValue) -> int:
        return self._observation.get_array()[card_value.value]

    def max_card_value(self) -> CardValue:
        array = self._observation.get_array()
        for i in range(12, -1, -1):
            if array[self._hand_begin_index + i] > 0:
                return CardValue(i)

    def play_cards(self, cards: PlayedCards):
        assert cards.count > 0

        array = self._observation.get_array()

        index = cards.value.value + self._hand_begin_index

        assert array[index] >= cards.count

        array[index] -= cards.count
