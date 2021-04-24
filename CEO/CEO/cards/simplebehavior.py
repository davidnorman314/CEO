from CEO.CEO.cards.player import *
from CEO.CEO.cards.hand import *


class SimpleBehaviorBase(PlayerBehaviorInterface):
    def pass_singles(self, hand: Hand, count: int) -> list[CardValue]:
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
