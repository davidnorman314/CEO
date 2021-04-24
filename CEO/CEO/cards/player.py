from CEO.CEO.cards.hand import *

class RoundState:
    """
    Class representing the state of the current round
    """
    def __init__(self):
        pass


class PlayerBehaviorInterface:
    """
    Interface implemented by objects that describe the plays made by a player
    """

    def pass_cards(self, hand: Hand, count: int) -> list[CardValue]:
        """
        Called to find out which cards are passed from a player on the
        high side of the table. This is not called for players on the low
        side.
        """
        pass

    def lead(self, hand: Hand, state: RoundState) -> CardValue:
        """
        Called to decide what the player should lead to start a trick.
        All cards of the given value will be lead.
        """
        pass

    def play_on_trick(self, hand: Hand, cur_trick_value : CardValue, cur_trick_count: int, state: RoundState) -> CardValue:
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

    def __init__(self, name : str, behavior: PlayerBehaviorInterface):
        self.name = name
        self.behavoir = behavior

    def __str__(self):
        return "Player " + self.name
