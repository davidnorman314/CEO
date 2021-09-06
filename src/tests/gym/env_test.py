import pytest
import random as random
from CEO.cards.eventlistener import EventListenerInterface, PrintAllEventListener
import CEO.cards.deck as deck
from CEO.cards.hand import *
from CEO.cards.simplebehavior import SimpleBehaviorBase
import CEO.cards.round as rd
import CEO.cards.player as player
from gym_ceo.envs.seat_ceo_env import SeatCEOEnv
from stable_baselines3.common.env_checker import check_env

from gym_ceo.envs.actions import Actions


class MockPlayerBehavior(player.PlayerBehaviorInterface, SimpleBehaviorBase):
    value_to_play: list[CardValue]
    to_play_next_index: int

    def __init__(self):
        self.value_to_play = []
        self.to_play_next_index = 0
        self.is_reinforcement_learning = False

    def pass_cards(self, hand: Hand, count: int) -> list[CardValue]:
        return self.pass_singles(hand, count)

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


def test_SeatCEOEnv_check_env():
    """
    Test using the Gym check_env
    """

    listener = EventListenerInterface()
    listener = PrintAllEventListener()

    print("Checking SeatCEOEnv. Seed 0")
    random.seed(0)
    env = SeatCEOEnv(listener=listener)
    check_env(env, True, True)

    print("Checking SeatCEOEnv. Seed 1")
    random.seed(1)
    env = SeatCEOEnv(listener=listener)
    check_env(env, True, True)

    print("Checking SeatCEOEnv. Seed 2")
    random.seed(2)
    env = SeatCEOEnv(listener=listener)
    check_env(env, True, True)


def test_SeatCEOEnv_Passing():
    """
    Test the environment that models a player in the the CEO seat. Here we test that passing
    cards at the beginning of the round happens. For other tests, see below where passing is
    disabled.
    """

    # Create CardValue objects for ease of use later
    cv0 = CardValue(0)
    cv1 = CardValue(1)
    cv2 = CardValue(2)
    cv3 = CardValue(3)
    cv4 = CardValue(4)
    cv5 = CardValue(5)
    cv6 = CardValue(6)
    cv7 = CardValue(7)

    # Make the hands.
    hand1 = Hand()
    hand1.add_cards(cv0, 1)
    hand1.add_cards(cv1, 1)
    hand1.add_cards(cv3, 2)

    hand2 = Hand()
    hand2.add_cards(cv1, 1)
    hand2.add_cards(cv2, 1)
    hand2.add_cards(cv4, 2)

    hand3 = Hand()
    hand3.add_cards(cv0, 1)
    hand3.add_cards(cv2, 1)
    hand3.add_cards(cv5, 2)
    hand3.add_cards(cv6, 2)

    hand4 = Hand()
    hand4.add_cards(cv1, 1)
    hand4.add_cards(cv2, 2)
    hand4.add_cards(cv3, 1)
    hand4.add_cards(cv7, 2)

    hands = [hand1, hand2, hand3, hand4]

    # Make the players. These aren't used.
    behavior2 = MockPlayerBehavior()
    behavior3 = MockPlayerBehavior()
    behavior4 = MockPlayerBehavior()

    behaviors = [behavior2, behavior3, behavior4]

    env = SeatCEOEnv(
        num_players=4,
        behaviors=behaviors,
        hands=hands,
        listener=PrintAllEventListener(),
    )

    observation = env.reset()

    assert observation[env.obs_index_hand_cards + 0] == 0
    assert observation[env.obs_index_hand_cards + 1] == 0
    assert observation[env.obs_index_hand_cards + 7] == 2


def test_SeatCEOEnv_NoPassing():
    """
    Test the environment that models a player in the the CEO seat. Here we disable passing
    to make the tests easier.
    """

    # Create CardValue objects for ease of use later
    cv0 = CardValue(0)
    cv1 = CardValue(1)
    cv2 = CardValue(2)
    cv3 = CardValue(3)
    cv4 = CardValue(4)
    cv5 = CardValue(5)

    # Make the hands. Note that we disable passing below
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
        num_players=4,
        behaviors=behaviors,
        hands=hands,
        listener=PrintAllEventListener(),
        skip_passing=True,
    )

    observation = env.reset()

    assert env.action_space.n == Actions.action_lead_count

    # Lead lowest
    observation, reward, done, info = env.step(Actions.play_lowest_num)

    assert env.action_space.n == Actions.action_play_count
    assert not done
    assert reward == 0

    observation, reward, done, info = env.step(Actions.play_highest_num)

    assert reward > 0
    assert done


def test_SeatCEOEnv_CanNotPlay_TwoTricks():
    """
    Test the environment that models a player in the the CEO seat.
    Here CEO has low cards so can't play on the final two tricks.
    """

    # Create CardValue objects for ease of use later
    cv0 = CardValue(0)
    cv1 = CardValue(1)
    cv2 = CardValue(2)
    cv3 = CardValue(3)
    cv4 = CardValue(4)
    cv5 = CardValue(5)
    cv6 = CardValue(6)
    cv7 = CardValue(7)

    # Make the hands
    hand1 = Hand()
    hand1.add_cards(cv0, 1)
    hand1.add_cards(cv1, 2)

    hand2 = Hand()
    hand2.add_cards(cv1, 1)
    hand2.add_cards(cv3, 2)
    hand2.add_cards(cv7, 1)

    hand3 = Hand()
    hand3.add_cards(cv2, 1)
    hand3.add_cards(cv4, 2)
    hand3.add_cards(cv5, 1)

    hand4 = Hand()
    hand4.add_cards(cv3, 1)
    hand4.add_cards(cv2, 2)
    hand4.add_cards(cv6, 1)

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
    # CEO must pass
    behavior2.value_to_play.append(cv3)
    behavior3.value_to_play.append(cv4)

    behavior3.value_to_play.append(cv5)
    behavior4.value_to_play.append(cv6)
    behavior2.value_to_play.append(cv7)
    # All players are out, so CEO drops

    behaviors = [behavior2, behavior3, behavior4]

    env = SeatCEOEnv(
        num_players=4,
        behaviors=behaviors,
        hands=hands,
        listener=PrintAllEventListener(),
        skip_passing=True,
    )

    observation = env.reset()

    # Lead lowest
    observation, reward, done, info = env.step(Actions.play_lowest_num)

    assert reward < 0
    assert done


def test_SeatCEOEnv_CanNotPlay_ThreeTricks():
    """
    Test the environment that models a player in the the CEO seat.
    Here CEO has low cards so can't play on the final three tricks.
    """

    # Create CardValue objects for ease of use later
    cv0 = CardValue(0)
    cv1 = CardValue(1)
    cv2 = CardValue(2)
    cv3 = CardValue(3)
    cv4 = CardValue(4)
    cv5 = CardValue(5)
    cv6 = CardValue(6)
    cv7 = CardValue(7)
    cv8 = CardValue(8)

    # Make the hands
    hand1 = Hand()
    hand1.add_cards(cv0, 1)
    hand1.add_cards(cv1, 2)

    hand2 = Hand()
    hand2.add_cards(cv1, 1)
    hand2.add_cards(cv3, 2)
    hand2.add_cards(cv7, 1)
    hand2.add_cards(cv6, 3)

    hand3 = Hand()
    hand3.add_cards(cv2, 1)
    hand3.add_cards(cv4, 2)
    hand3.add_cards(cv5, 1)
    hand3.add_cards(cv7, 3)

    hand4 = Hand()
    hand4.add_cards(cv3, 1)
    hand4.add_cards(cv2, 2)
    hand4.add_cards(cv6, 1)
    hand4.add_cards(cv8, 3)

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
    # CEO must pass
    behavior2.value_to_play.append(cv3)
    behavior3.value_to_play.append(cv4)

    behavior3.value_to_play.append(cv5)
    behavior4.value_to_play.append(cv6)
    behavior2.value_to_play.append(cv7)

    behavior2.value_to_play.append(cv6)
    behavior3.value_to_play.append(cv7)
    behavior4.value_to_play.append(cv8)

    behaviors = [behavior2, behavior3, behavior4]

    env = SeatCEOEnv(
        num_players=4,
        behaviors=behaviors,
        hands=hands,
        listener=PrintAllEventListener(),
        skip_passing=True,
    )

    observation = env.reset()

    # Lead lowest
    observation, reward, done, info = env.step(Actions.play_lowest_num)

    assert done
    assert reward < 0
