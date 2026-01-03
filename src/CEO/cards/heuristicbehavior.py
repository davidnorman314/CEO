from CEO.cards.hand import CardValue, Hand
from CEO.cards.player import PlayerBehaviorInterface, RoundState
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
        playable_list = self.get_playable_cards(hand, cur_trick_value, cur_trick_count)

        # Remove playable cards that would break up a set and that aren't aces.
        playable_list = [p for p in playable_list if p.cv.is_ace() or p.count_matches]

        if len(playable_list) == 0:
            return None

        # Find the lowest group that can be played on a trick without breaking up a
        # group.
        lowest_can_play = playable_list[0].cv.value

        higher_players_left = (
            starting_position > 0 and starting_position < player_position
        )

        # The number of groups of cards that can't be played on the trick
        groups_lower = sum(
            map(lambda i: hand.count(CardValue(i)) > 0, range(lowest_can_play))
        )

        # The number of groups of cards that can be played on the trick
        groups_higher = sum(
            map(lambda i: hand.count(CardValue(i)) > 0, range(lowest_can_play, 13))
        )

        if higher_players_left:
            if groups_lower >= groups_higher and lowest_can_play < 12:
                return None
            else:
                return CardValue(lowest_can_play)

        # See if we should play an ace
        highest_playable = playable_list[-1]
        if highest_playable.cv.is_ace():
            return highest_playable.cv

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
            return bool(cvi == 12 and cur_trick_count == 1)

        return True
