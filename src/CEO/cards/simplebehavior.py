from CEO.cards.player import *
from CEO.cards.hand import *


class PlayableCard:
    cv: CardValue
    count_matches: bool
    breaks_pair: bool
    breaks_triple: bool
    breaks_quadruple: bool
    breaks_large: bool

    def __init__(self, cv: CardValue, hand_card_count: int, trick_card_count: int):
        self.cv = cv

        self.count_matches = False
        self.breaks_pair = False
        self.breaks_triple = False
        self.breaks_quadruple = False
        self.breaks_large = False

        if hand_card_count == trick_card_count:
            self.count_matches = True
        elif hand_card_count == 2:
            self.breaks_pair = True
        elif hand_card_count == 3:
            self.breaks_triple = True
        elif hand_card_count == 3:
            self.breaks_quadruple = True
        elif hand_card_count == 3:
            self.breaks_large = True


class SimpleBehaviorBase:
    def get_playable_cards(
        self, hand: Hand, cur_trick_value: CardValue, trick_card_count: int
    ) -> list[PlayableCard]:
        """
        Returns a list of all cards that can be played on the trick.
        """

        assert cur_trick_value is not None

        ret = []
        for i in range(cur_trick_value.value + 1, 13):
            cv = CardValue(i)
            hand_count = hand.count(cv)
            if hand_count >= trick_card_count:
                ret.append(PlayableCard(cv, hand_count, trick_card_count))

        return ret

    def get_leadable_cards(self, hand: Hand) -> list[PlayableCard]:
        """
        Returns a list of all cards that can be lead, i.e., all cards in the trick.
        """

        ret = []
        for i in range(13):
            cv = CardValue(i)
            hand_count = hand.count(cv)
            if hand_count > 0:
                ret.append(PlayableCard(cv, hand_count, hand_count))

        return ret

    def pass_singles(self, hand: Hand, count: int) -> list[CardValue]:
        """
        Function that passes the lowest singles in a hand.
        If there aren't enough singles, then a card from the lowest
        pair will be passed.
        """
        try:
            return self._pass_singles_internal(hand, count)
        except AssertionError:
            # If there is an assertion, print out the state so that we can debug it.
            print("Can't find", count, "cards to pass:", hand)

            raise

    def _pass_singles_internal(self, hand: Hand, count: int) -> list[CardValue]:
        """
        Function that passes the lowest singles in a hand.
        If there aren't enough singles, then a card from the lowest
        pair will be passed.
        """
        ret = []

        triple_count = 0
        quad_count = 0
        pair_count = 0
        for i in range(13):
            cv = CardValue(i)
            ct = hand.count(cv)

            if ct == 2:
                pair_count += 1
                continue
            elif ct == 3:
                triple_count += 1
                continue
            elif ct == 4:
                quad_count += 1
                continue
            elif hand.count(cv) != 1:
                continue

            ret.append(cv)

            if len(ret) == count:
                return ret

        single_count = len(ret)

        # We don't have enough singles, so we need to pass one or more
        # cards from the lowest pair.
        seen_lowest_pair = False
        seen_second_lowest_pair = False
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
                elif not seen_second_lowest_pair and single_count == 0:
                    ret.append(cv)
                    seen_second_lowest_pair = True
                else:
                    continue
            else:
                assert hand.count(cv) == 1

                ret.append(cv)

            if len(ret) == count:
                return ret

        # See if we should just pass the lowest triple
        if single_count == 0 and triple_count > 0 and count == 3:
            ret = []

            for i in range(13):
                cv = CardValue(i)
                if hand.count(cv) != 3:
                    continue

                ret = [cv] * count
                break

            return ret

        # We don't have enough singles and don't have a pair, so we need to pass the singles and
        # cards from the lowest other group.
        seen_lowest_large_group = False
        ret = []

        for i in range(13):
            cv = CardValue(i)
            if hand.count(cv) == 1:
                ret.append(cv)
            elif hand.count(cv) == 0:
                pass
            else:
                if not seen_lowest_large_group:
                    ret.extend([cv] * (count - single_count))

                    seen_lowest_large_group = True
                else:
                    continue

            if len(ret) == count:
                return ret

        assert "shouldn't be here" == ""

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


class BasicBehavior(PlayerBehaviorInterface, SimpleBehaviorBase):
    """
    Class implementing a simple, non-optimal behavior
    """

    def __init__(self):
        self.is_reinforcement_learning = False

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
        return self.play_lowest_or_pass(hand, cur_trick_value, cur_trick_count, state)