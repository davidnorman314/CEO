from CEO.cards.player import *
from CEO.cards.hand import *
from CEO.cards.simplebehavior import SimpleBehaviorBase


class Actions(SimpleBehaviorBase):
    def play_lowest(
        self, hand: Hand, cur_trick_value: CardValue, cur_trick_count: int
    ) -> CardValue:

        playable_list = self.get_playable_cards(hand, cur_trick_value, cur_trick_count)

        if len(playable_list) == 0:
            return None

        return playable_list[0].cv

    def play_second_lowest(
        self, hand: Hand, cur_trick_value: CardValue, cur_trick_count: int
    ) -> CardValue:

        playable_list = self.get_playable_cards(hand, cur_trick_value, cur_trick_count)

        if len(playable_list) == 0:
            return None

        if len(playable_list) == 1:
            return playable_list[0].cv

        return playable_list[1].cv

    def play_highest(
        self, hand: Hand, cur_trick_value: CardValue, cur_trick_count: int
    ) -> CardValue:

        playable_list = self.get_playable_cards(hand, cur_trick_value, cur_trick_count)

        if len(playable_list) == 0:
            return None

        return playable_list[-1].cv
