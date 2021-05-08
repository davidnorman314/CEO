from CEO.cards.player import *
from CEO.cards.hand import *
from CEO.cards.simplebehavior import SimpleBehaviorBase


class HeuristicBehavior(SimpleBehaviorBase, PlayerBehaviorInterface):
    """
    Class implementing a player that uses certain heuristics
    """

    def pass_cards(self, hand: Hand, count: int) -> list[CardValue]:
        return self.pass_singles(hand, count)

    def lead(self, player_position: int, hand: Hand, state: RoundState) -> CardValue:
        return self.lead_lowest(hand, state)

    def play_on_trick(
        self,
        starting_position: int,
        player_position: int,
        hand: Hand,
        cur_trick_value: CardValue,
        cur_trick_count: int,
        state: RoundState,
    ) -> CardValue:

        # We have to pass if the current value is an ace
        if cur_trick_value == CardValue(12):
            return None

        higher_players_left = starting_position > 0 and starting_position < player_position

        # The number of groups of cards that can't be played on the trick
        groups_lower = sum(
            map(lambda i: hand.count(CardValue(i)) > 0, range(cur_trick_value.value + 1))
        )

        # The number of groups of cards that can be played on the trick
        groups_higher = sum(
            map(lambda i: hand.count(CardValue(i)) > 0, range(cur_trick_value.value + 1, 13))
        )

        if higher_players_left:
            if groups_lower >= groups_higher:
                return None
            else:
                return self.play_lowest_or_pass(hand, cur_trick_value, cur_trick_count, state)

        return self.play_lowest_or_pass(hand, cur_trick_value, cur_trick_count, state)
