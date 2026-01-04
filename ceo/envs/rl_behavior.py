from ceo.game.hand import CardValue, Hand
from ceo.game.player import PlayerBehaviorInterface
from ceo.game.round import RoundState
from ceo.game.simplebehavior import SimpleBehaviorBase


class RLBehavior(PlayerBehaviorInterface, SimpleBehaviorBase):
    """
    Class used for RL behavior
    """

    def __init__(self):
        self.is_reinforcement_learning = True

    def pass_cards(self, hand: Hand, count: int) -> list[CardValue]:
        return self.pass_singles(hand, count)

    def lead(self, player_position: int, hand: Hand, state: RoundState) -> CardValue:
        assert not "This should not be called"

    def play_on_trick(
        self,
        starting_position: int,
        player_position: int,
        hand: Hand,
        cur_trick_value: CardValue,
        cur_trick_count: int,
        state: RoundState,
    ) -> CardValue:
        assert not "This should not be called"
