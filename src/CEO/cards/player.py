from abc import ABC, abstractmethod
from CEO.cards.hand import *

from typing import MutableSequence


class RoundState:
    """
    Class representing the state of the current round
    """

    cards_remaining: MutableSequence[int]

    def __init__(self):
        pass

    def initialize(self, players, hands: list[Hand]):
        self.cards_remaining = array.array("i", [hand.card_count() for hand in hands])


class PlayerBehaviorInterface(ABC):
    """
    Interface implemented by objects that describe the plays made by a player
    """

    is_reinforcement_learning: bool

    @abstractmethod
    def pass_cards(self, hand: Hand, count: int) -> list[CardValue]:
        """
        Called to find out which cards are passed from a player on the
        high side of the table. This is not called for players on the low
        side.
        """
        pass

    @abstractmethod
    def lead(self, player_position: int, hand: Hand, state: RoundState) -> CardValue:
        """
        Called to decide what the player should lead to start a trick.
        All cards of the given value will be lead.
        """
        pass

    @abstractmethod
    def play_on_trick(
        self,
        starting_position: int,
        player_position: int,
        hand: Hand,
        cur_trick_value: CardValue,
        cur_trick_count: int,
        state: RoundState,
    ) -> CardValue:
        """
        Called to decide what the player should play on the given trick.
        Returns the value of the card(s) that should be played.
        If the player passes, then None should be returned.
        """
        pass


class Player:
    """
    Class describing a player in a CEO game
    """

    def __init__(self, name: str, behavior: PlayerBehaviorInterface):
        self.name = name
        self.behavoir = behavior

    def __str__(self):
        return "Player " + self.name

    def __repr__(self):
        return "Player " + self.name
