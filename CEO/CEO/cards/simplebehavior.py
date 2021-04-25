from CEO.CEO.cards.player import *
from CEO.CEO.cards.hand import *


class SimpleBehaviorBase(PlayerBehaviorInterface):
    def pass_singles(self, hand: Hand, count: int) -> list[CardValue]:
        """
        Function that passes the three lowest singles in a hand.
        If there aren't enough singles, then a card from the lowest
        pair will be passed.
        """
        ret = []

        for i in range(13):
            cv = CardValue(i)
            if hand.count(cv) != 1:
                continue

            ret.append(cv)

            if len(ret) == count:
                return ret

        # We don't have enough singles, so we need to pass one or more
        # cards from the lowest pair.
        seen_lowest_pair = False
        ret = []

        for i in range(13):
            cv = CardValue(i)
            if hand.count(cv) > 2 or hand.count(cv) == 0:
                continue
            elif hand.count(cv) == 2:
                if not seen_lowest_pair:
                    ret.append(cv)

                    if len(ret) < count:
                        ret.append(cv)

                    seen_lowest_pair = True
                else:
                    continue
            else:
                assert hand.count(cv) == 1

                ret.append(cv)

            if len(ret) == count:
                return ret

    def play_lowest_or_pass(
        self, hand: Hand, cur_trick_value: CardValue, cur_trick_count: int, state: RoundState
    ) -> CardValue:
        """
        Method that either playes the lowest card possible or passes.
        It will break up sets, i.e., break up a pair to play a single.
        """
        for i in range(cur_trick_value.value + 1, 13):
            if hand.count(CardValue(i)) >= cur_trick_count:
                return CardValue(i)

        # We can't play on the trick
        return None

    def lead_lowest(self, hand: Hand, state: RoundState) -> CardValue:
        """
        Method that leads the hand's lowest card
        """

        for i in range(13):
            if hand.count(CardValue(i)) > 0:
                return CardValue(i)

        assert "Hand doesn't have cards" == ""
