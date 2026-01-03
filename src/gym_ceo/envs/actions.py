from enum import Enum

from gymnasium.spaces import Discrete

from CEO.cards.hand import HandInterface
from CEO.cards.player import *
from CEO.cards.simplebehavior import SimpleBehaviorBase


class ActionEnum(Enum):
    PLAY_HIGHEST_NUM = 0
    PLAY_SECOND_LOWEST_WITHOUT_BREAK_NUM = 1
    PLAY_LOWEST_WITHOUT_BREAK_NUM = 2
    PASS_ON_TRICK_NUM = 3


class Actions(SimpleBehaviorBase):
    max_action_count = len(ActionEnum)

    action_play_one_legal_count = 2

    def lead(self):
        pass

    def play(
        self,
        hand: HandInterface,
        cur_trick_value: CardValue,
        cur_trick_count: int,
        action_number: int,
    ) -> CardValue:
        if action_number == ActionEnum.PASS_ON_TRICK_NUM:
            if cur_trick_value is None:
                print("Action pass for lead")
                print(" Hand", hand)
                assert cur_trick_value is not None
                assert cur_trick_count is not None

            return self.pass_on_trick(hand, cur_trick_value, cur_trick_count)
        elif action_number == ActionEnum.PLAY_LOWEST_WITHOUT_BREAK_NUM:
            # ret = self.play_lowest(hand, cur_trick_value, cur_trick_count)
            ret = self.play_lowest_without_breaking_sets(
                hand, cur_trick_value, cur_trick_count
            )
        elif action_number == ActionEnum.PLAY_SECOND_LOWEST_WITHOUT_BREAK_NUM:
            # ret = self.play_second_lowest(hand, cur_trick_value, cur_trick_count)
            ret = self.play_second_lowest_without_breaking_sets(
                hand, cur_trick_value, cur_trick_count
            )
        elif action_number == ActionEnum.PLAY_HIGHEST_NUM:
            ret = self.play_highest(hand, cur_trick_value, cur_trick_count)
        else:
            print("invalid action ", str(action_number))
            assert "invalid action " + str(action_number) == ""

        return ret

    def play_lowest(
        self, hand: HandInterface, cur_trick_value: CardValue, cur_trick_count: int
    ) -> CardValue:
        if cur_trick_value is None:
            playable_list = self.get_leadable_cards(hand)
        else:
            playable_list = self.get_playable_cards(
                hand, cur_trick_value, cur_trick_count
            )

        if len(playable_list) == 0:
            return None

        return playable_list[0].cv

    def play_lowest_without_breaking_sets(
        self, hand: HandInterface, cur_trick_value: CardValue, cur_trick_count: int
    ) -> CardValue:
        if cur_trick_value is None:
            return self.play_lowest(hand, cur_trick_value, cur_trick_count)
        else:
            playable_list = self.get_playable_cards(
                hand, cur_trick_value, cur_trick_count
            )

        if len(playable_list) == 0:
            return None

        for value in playable_list:
            if hand.count(value.cv) == cur_trick_count:
                return value.cv

        return None

    def play_second_lowest(
        self, hand: HandInterface, cur_trick_value: CardValue, cur_trick_count: int
    ) -> CardValue:
        if cur_trick_value is None:
            playable_list = self.get_leadable_cards(hand)
        else:
            playable_list = self.get_playable_cards(
                hand, cur_trick_value, cur_trick_count
            )

        if len(playable_list) == 0:
            return None

        if len(playable_list) == 1:
            return playable_list[0].cv

        return playable_list[1].cv

    def play_second_lowest_without_breaking_sets(
        self, hand: HandInterface, cur_trick_value: CardValue, cur_trick_count: int
    ) -> CardValue:
        """
        Play the second lowest card without breaking up any sets. If there is a single value
        that can be played, then play it.
        """

        if cur_trick_value is None:
            return self.play_second_lowest(hand, cur_trick_value, cur_trick_count)
        else:
            playable_list = self.get_playable_cards(
                hand, cur_trick_value, cur_trick_count
            )

        if len(playable_list) == 0:
            return None

        found_lowest = False
        for value in playable_list:
            if hand.count(value.cv) == cur_trick_count:
                if not found_lowest:
                    found_lowest = True
                else:
                    return value.cv

        if found_lowest:
            # We found a value to play, but not two value to play. Play the one value.
            for value in playable_list:
                if hand.count(value.cv) == cur_trick_count:
                    return value.cv

        return None

    def play_highest(
        self, hand: HandInterface, cur_trick_value: CardValue, cur_trick_count: int
    ) -> CardValue:
        if cur_trick_value is None:
            playable_list = self.get_leadable_cards(hand)
        else:
            playable_list = self.get_playable_cards(
                hand, cur_trick_value, cur_trick_count
            )

        if len(playable_list) == 0:
            return None

        return playable_list[-1].cv

    def pass_on_trick(
        self, hand: HandInterface, cur_trick_value: CardValue, cur_trick_count: int
    ) -> CardValue:
        return None


class CEOActionSpace(Discrete):
    actions: list[ActionEnum]

    _actions_obj: Actions

    def __init__(self, actions: list[int]):
        super(CEOActionSpace, self).__init__(len(actions))

        self.actions = actions
        self._actions_obj = Actions()

    def find_full_action(self, full_action: int) -> int:
        return self.actions.index(full_action)

    def card_to_play(
        self,
        hand: HandInterface,
        cur_trick_value: CardValue,
        cur_trick_count: int,
        action: int,
    ) -> CardValue:
        full_action = self.actions[action]

        cv = self._actions_obj.play(hand, cur_trick_value, cur_trick_count, full_action)

        return cv

    def __eq__(self, other):
        if not super(CEOActionSpace, self).__eq__(other):
            return False

        return self.actions == other.actions


class ActionSpaceFactory(SimpleBehaviorBase):
    """Class that calculates the correct action space based on the hand and trick"""

    action_space_lead = CEOActionSpace(
        [
            ActionEnum.PLAY_HIGHEST_NUM,
            ActionEnum.PLAY_SECOND_LOWEST_WITHOUT_BREAK_NUM,
            ActionEnum.PLAY_LOWEST_WITHOUT_BREAK_NUM,
        ]
    )
    action_space_one_legal_lead = CEOActionSpace(
        [
            ActionEnum.PLAY_HIGHEST_NUM,
        ]
    )
    action_space_two_legal_lead = CEOActionSpace(
        [
            ActionEnum.PLAY_HIGHEST_NUM,
            ActionEnum.PLAY_LOWEST_WITHOUT_BREAK_NUM,
        ]
    )
    action_space_play = CEOActionSpace(
        [
            ActionEnum.PLAY_HIGHEST_NUM,
            ActionEnum.PLAY_SECOND_LOWEST_WITHOUT_BREAK_NUM,
            ActionEnum.PLAY_LOWEST_WITHOUT_BREAK_NUM,
            ActionEnum.PASS_ON_TRICK_NUM,
        ]
    )
    action_space_one_legal_play = CEOActionSpace(
        [
            ActionEnum.PLAY_HIGHEST_NUM,
            ActionEnum.PASS_ON_TRICK_NUM,
        ]
    )
    action_space_two_legal_play = CEOActionSpace(
        [
            ActionEnum.PLAY_HIGHEST_NUM,
            ActionEnum.PLAY_LOWEST_WITHOUT_BREAK_NUM,
            ActionEnum.PASS_ON_TRICK_NUM,
        ]
    )

    def __init__(self):
        pass

    def default_lead(self):
        return self.action_space_lead

    def create_lead(self, hand: HandInterface):
        # Count the number of playable card values.
        playable_card_values = 0
        for cv in range(13):
            if hand.count(CardValue(cv)) > 0:
                playable_card_values += 1

        if playable_card_values == 1:
            return self.action_space_one_legal_lead
        elif playable_card_values == 2:
            return self.action_space_two_legal_lead
        else:
            return self.action_space_lead

    def create_play(
        self, hand: HandInterface, cur_trick_value: CardValue, cur_trick_count: int
    ):
        """Create the action space where the player will play on the given trick"""
        playable_cards = self.get_playable_cards(hand, cur_trick_value, cur_trick_count)
        playable_card_count = len(playable_cards)

        # See if we must pass, i.e., there is no choice of action or if there
        # is only one legal play.
        if playable_card_count == 0:
            return None
        elif playable_card_count == 1:
            return self.action_space_one_legal_play

        highest_value = playable_cards[-1].cv

        # Filter out low cards that if played would break up a set.
        playable_no_break_card_values = [
            playable_card
            for playable_card in playable_cards
            if playable_card.count_matches or playable_card.cv == highest_value
        ]

        updated_playable_card_count = len(playable_no_break_card_values)

        if updated_playable_card_count == 1:
            return self.action_space_one_legal_play
        elif updated_playable_card_count == 2:
            return self.action_space_two_legal_play
        else:
            return self.action_space_play


class CardActionSpace(Discrete):
    def __init__(self, card_count: int):
        super(CardActionSpace, self).__init__(card_count)

    def card_to_play(
        self,
        hand: HandInterface,
        cur_trick_value: CardValue,
        cur_trick_count: int,
        action: int,
    ) -> CardValue:
        if cur_trick_value is None:
            # The action is to lead
            playable_card_values = -1
            for cv in range(13):
                if hand.count(CardValue(cv)) > 0:
                    playable_card_values += 1

                    if playable_card_values == action:
                        return CardValue(cv)

            assert "Action to large when leading" == ""
        else:
            # The action is to play on a trick
            playable_cards = hand.get_playable_cards(cur_trick_value, cur_trick_count)

            if action == len(playable_cards):
                # Pass
                return None

            return playable_cards[action].cv

    def __eq__(self, other):
        if not super(CardActionSpace, self).__eq__(other):
            return False

        return self.n == other.n


class CardActionSpaceFactory(SimpleBehaviorBase):
    """Class that calculates the correct card action space based on the hand and trick. Here
    the action space corresponds to playing a given card in the hand"""

    _spaces: list[CardActionSpace]

    def __init__(self):
        self._spaces = []

        for cv in range(14):
            if cv == 0:
                self._spaces.append(None)
            else:
                self._spaces.append(CardActionSpace(cv))

    def default_lead(self):
        return self._spaces[-1]

    def create_lead(self, hand: HandInterface):
        """Create the action space where the player will lead"""
        # Count the number of playable card values.
        playable_card_values = 0
        for cv in range(13):
            if hand.count(CardValue(cv)) > 0:
                playable_card_values += 1

        return self._spaces[playable_card_values]

    def create_play(
        self, hand: HandInterface, cur_trick_value: CardValue, cur_trick_count: int
    ):
        """Create the action space where the player will play on the given trick"""
        playable_cards = self.get_playable_cards(hand, cur_trick_value, cur_trick_count)

        return self._spaces[len(playable_cards) + 1]


class AllCardActionSpace(Discrete):
    """Action space that always has 14 actions: one for each card and one for pass.
    If an action is not valid, e.g., the card isn't in the hand, then a negative reward
    is returned."""

    _pass_action = 13

    def __init__(self):
        super(AllCardActionSpace, self).__init__(14)

    def _find_largest_card(self, hand: HandInterface):
        for cv in reversed(range(13)):
            if hand.count(CardValue(cv)) > 0:
                return CardValue(cv)

        assert "Should not get here" == ""

    def _get_invalid_action_penalty(self, hand: HandInterface):
        return -(2.0 + 8 * (hand.card_count() / 13.0))

    def card_to_play(
        self,
        hand: HandInterface,
        cur_trick_value: CardValue,
        cur_trick_count: int,
        action: int,
    ) -> CardValue:
        if cur_trick_value is None:
            # The action is to lead

            if action == self._pass_action:
                # The action is to pass, which isn't valid.
                return (None, self._get_invalid_action_penalty(hand))

            cv = CardValue(action)
            hand_card_count = hand.count(cv)

            if hand_card_count == 0:
                # We don't have this card in our hand, so this is an invalid action.
                return (None, self._get_invalid_action_penalty(hand))

            return cv

        else:
            # The action is to play on a trick

            # See if we should pass
            if action == self._pass_action:
                return None

            cv = CardValue(action)
            hand_card_count = hand.count(cv)

            if cv.value <= cur_trick_value.value:
                # The card is too small, so this is an invalid action.
                return (None, self._get_invalid_action_penalty(hand))

            if hand_card_count < cur_trick_count:
                # We don't have enough cards to play, so this is an invalid action.
                return (None, self._get_invalid_action_penalty(hand))

            return cv

    def __eq__(self, other):
        if not super(AllCardActionSpace, self).__eq__(other):
            return False

        return self.n == other.n


class AllCardActionSpaceFactory(SimpleBehaviorBase):
    """Action space factory for the AllCardActionSpace. It always returns the same space."""

    _space: AllCardActionSpace

    def __init__(self):
        self._space = AllCardActionSpace()

    def default_lead(self):
        return self._space

    def create_lead(self, hand: HandInterface):
        """Create the action space where the player will lead"""
        return self._space

    def create_play(
        self, hand: HandInterface, cur_trick_value: CardValue, cur_trick_count: int
    ):
        """Create the action space where the player will play on the given trick"""
        return self._space
