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

        # Find the lowest group that can be played on a trick
        lowest_can_play = None
        for cvi in range(13):
            if self._can_play_card(cvi, hand, cur_trick_value, cur_trick_count):
                lowest_can_play = cvi
                break

        if lowest_can_play is None:
            return None

        # The number of groups of cards that can't be played on the trick
        groups_lower = sum(map(lambda i: hand.count(CardValue(i)) > 0, range(lowest_can_play)))

        # The number of groups of cards that can be played on the trick
        groups_higher = sum(map(lambda i: hand.count(CardValue(i)) > 0, range(lowest_can_play, 13)))

        if higher_players_left:
            if groups_lower >= groups_higher and lowest_can_play < 12:
                return None
            else:
                # return self.play_lowest_or_pass(hand, cur_trick_value, cur_trick_count, state)
                return CardValue(lowest_can_play)

        # return self.play_lowest_or_pass(hand, cur_trick_value, cur_trick_count, state)
        return CardValue(lowest_can_play)

    def _can_play_card(
        self, cvi: int, hand: Hand, cur_trick_value: CardValue, cur_trick_count: int
    ) -> bool:
        cv = CardValue(cvi)
        hand_count = hand.count(cv)

        if hand_count == 0 or hand_count < cur_trick_count:
            return False

        if cv.value <= cur_trick_value.value:
            return False

        # Don't break up groups, with some exceptions
        if hand_count > cur_trick_count:
            # Do break up aces to play on singles
            if cvi == 12 and cur_trick_count == 1:
                return True

            return False

        return True
