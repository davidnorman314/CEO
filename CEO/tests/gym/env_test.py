from CEO.cards.eventlistener import PrintAllEventListener
import pytest
import CEO.cards.deck as deck
from CEO.cards.hand import *
import CEO.cards.round as rd
import CEO.cards.player as player
from gym_ceo.envs.seat_ceo_env import SeatCEOEnv
from stable_baselines3.common.env_checker import check_env

from gym_ceo.envs.actions import Actions


class MockPlayerBehavior(player.PlayerBehaviorInterface):
    value_to_play: list[CardValue]
    to_play_next_index: int

    def __init__(self):
        self.value_to_play = []
        self.to_play_next_index = 0

    def pass_cards(self, hand: Hand, count: int) -> list[CardValue]:
        raise NotImplemented

    def lead(self, player_position: int, hand: Hand, state) -> CardValue:

        if len(self.value_to_play) <= self.to_play_next_index:
            assert "No more values to play" != ""

        ret = self.value_to_play[self.to_play_next_index]
        self.to_play_next_index += 1

        return ret

    def play_on_trick(
        self,
        starting_position: int,
        player_position: int,
        hand: Hand,
        cur_trick_value: CardValue,
        cur_trick_count: int,
        state: rd.RoundState,
    ) -> CardValue:

        if len(self.value_to_play) <= self.to_play_next_index:
            assert "No more values to play" != ""

        ret = self.value_to_play[self.to_play_next_index]
        self.to_play_next_index += 1

        return ret


def test_check_env():
    """
    Test using the Gym check_env
    """

    print("Checking SeatCEOEnv.")

    env = SeatCEOEnv()

    check_env(env, True, True)


def test_SeatCEOEnv():
    """
    Test the environment that models a player in the the CEO seat.
    """

    # Create CardValue objects for ease of use later
    cv0 = CardValue(0)
    cv1 = CardValue(1)
    cv2 = CardValue(2)
    cv3 = CardValue(3)
    cv4 = CardValue(4)
    cv5 = CardValue(5)

    # Make the hands
    hand1 = Hand()
    hand1.add_cards(cv0, 1)
    hand1.add_cards(cv3, 2)

    hand2 = Hand()
    hand2.add_cards(cv1, 1)
    hand2.add_cards(cv4, 2)
    hand2.add_cards(cv2, 1)

    hand3 = Hand()
    hand3.add_cards(cv2, 1)
    hand3.add_cards(cv5, 2)
    hand3.add_cards(cv0, 1)

    hand4 = Hand()
    hand4.add_cards(cv3, 1)
    hand4.add_cards(cv2, 2)
    hand4.add_cards(cv1, 1)

    hands = [hand1, hand2, hand3, hand4]

    # Make the players
    behavior2 = MockPlayerBehavior()
    behavior3 = MockPlayerBehavior()
    behavior4 = MockPlayerBehavior()

    # action: Lead lowest = cv0
    behavior2.value_to_play.append(cv1)
    behavior3.value_to_play.append(cv2)
    behavior4.value_to_play.append(cv3)

    behavior4.value_to_play.append(cv2)
    # action: Play highest = cv3
    behavior2.value_to_play.append(cv4)
    behavior3.value_to_play.append(cv5)

    behavior3.value_to_play.append(cv0)
    behavior4.value_to_play.append(cv1)
    behavior2.value_to_play.append(cv2)

    behaviors = [behavior2, behavior3, behavior4]

    env = SeatCEOEnv(
        num_players=4, behaviors=behaviors, hands=hands, listener=PrintAllEventListener()
    )

    observation = env.reset()

    observation, reward, done, info = env.step(Actions.play_lowest_num)

    assert not done
    assert reward == 0

    observation, reward, done, info = env.step(Actions.play_highest_num)

    assert reward > 0
    assert done
