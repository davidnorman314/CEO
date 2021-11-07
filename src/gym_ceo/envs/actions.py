from CEO.cards.player import *
from CEO.cards.hand import *
from CEO.cards.simplebehavior import SimpleBehaviorBase

from gym.spaces import Box, Discrete

from enum import Enum


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
        self, hand: Hand, cur_trick_value: CardValue, cur_trick_count: int, action_number: int
    ):
        if action_number == ActionEnum.PASS_ON_TRICK_NUM:
            if cur_trick_value is None:
                print("Action pass for lead")
                print(" Hand", hand)
                assert cur_trick_value is not None
                assert cur_trick_count is not None

            return self.pass_on_trick(hand, cur_trick_value, cur_trick_count)
        elif action_number == ActionEnum.PLAY_LOWEST_WITHOUT_BREAK_NUM:
            # ret = self.play_lowest(hand, cur_trick_value, cur_trick_count)
            ret = self.play_lowest_without_breaking_sets(hand, cur_trick_value, cur_trick_count)
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
        self, hand: Hand, cur_trick_value: CardValue, cur_trick_count: int
    ) -> CardValue:

        if cur_trick_value is None:
            playable_list = self.get_leadable_cards(hand)
        else:
            playable_list = self.get_playable_cards(hand, cur_trick_value, cur_trick_count)

        if len(playable_list) == 0:
            return None

        return playable_list[0].cv

    def play_lowest_without_breaking_sets(
        self, hand: Hand, cur_trick_value: CardValue, cur_trick_count: int
    ) -> CardValue:

        if cur_trick_value is None:
            return self.play_lowest(hand, cur_trick_value, cur_trick_count)
        else:
            playable_list = self.get_playable_cards(hand, cur_trick_value, cur_trick_count)

        if len(playable_list) == 0:
            return None

        for value in playable_list:
            if hand.count(value.cv) == cur_trick_count:
                return value.cv

        return None

    def play_second_lowest(
        self, hand: Hand, cur_trick_value: CardValue, cur_trick_count: int
    ) -> CardValue:

        if cur_trick_value is None:
            playable_list = self.get_leadable_cards(hand)
        else:
            playable_list = self.get_playable_cards(hand, cur_trick_value, cur_trick_count)

        if len(playable_list) == 0:
            return None

        if len(playable_list) == 1:
            return playable_list[0].cv

        return playable_list[1].cv

    def play_second_lowest_without_breaking_sets(
        self, hand: Hand, cur_trick_value: CardValue, cur_trick_count: int
    ) -> CardValue:
        """
        Play the second lowest card without breaking up any sets. If there is a single value
        that can be played, then play it.
        """

        if cur_trick_value is None:
            return self.play_second_lowest(hand, cur_trick_value, cur_trick_count)
        else:
            playable_list = self.get_playable_cards(hand, cur_trick_value, cur_trick_count)

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
        self, hand: Hand, cur_trick_value: CardValue, cur_trick_count: int
    ) -> CardValue:

        if cur_trick_value is None:
            playable_list = self.get_leadable_cards(hand)
        else:
            playable_list = self.get_playable_cards(hand, cur_trick_value, cur_trick_count)

        if len(playable_list) == 0:
            return None

        return playable_list[-1].cv

    def pass_on_trick(
        self, hand: Hand, cur_trick_value: CardValue, cur_trick_count: int
    ) -> CardValue:
        return None


class CEOActionSpace(Discrete):
    actions: list[ActionEnum]

    def __init__(self, actions: list[int]):
        super(CEOActionSpace, self).__init__(len(actions))

        self.actions = actions

    def find_full_action(self, full_action: int) -> int:
        return self.actions.index(full_action)

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

    def create_lead(self, hand: Hand):
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

    def create_play(self, hand: Hand, cur_trick_value: CardValue, cur_trick_count: int):
        """Create the action space where the player will play on the given trick"""
        playable_card_values = len(self.get_playable_cards(hand, cur_trick_value, cur_trick_count))

        # See if we must pass, i.e., there is no choice of action
        if playable_card_values == 0:
            return None
        elif playable_card_values == 1:
            return self.action_space_one_legal_play
        elif playable_card_values == 2:
            return self.action_space_two_legal_play
        else:
            return self.action_space_play
