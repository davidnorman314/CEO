from CEO.cards.player import *
from CEO.cards.hand import *
from CEO.cards.simplebehavior import SimpleBehaviorBase


class Actions(SimpleBehaviorBase):

    play_highest_num = 0
    play_second_lowest_num = 1
    play_lowest_num = 2
    pass_on_trick_num = 3

    action_lead_count = 3
    action_play_count = 4

    def lead(self):
        pass

    def play(
        self, hand: Hand, cur_trick_value: CardValue, cur_trick_count: int, action_number: int
    ):
        if action_number == self.pass_on_trick_num:
            if cur_trick_value is None:
                print("Action pass for lead")
                print(" Hand", hand)
                assert cur_trick_value is not None
                assert cur_trick_count is not None

            return self.pass_on_trick(hand, cur_trick_value, cur_trick_count)
        elif action_number == self.play_lowest_num:
            ret = self.play_lowest(hand, cur_trick_value, cur_trick_count)
        elif action_number == self.play_second_lowest_num:
            ret = self.play_second_lowest(hand, cur_trick_value, cur_trick_count)
        elif action_number == self.play_highest_num:
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
