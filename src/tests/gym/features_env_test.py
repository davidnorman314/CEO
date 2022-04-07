import pytest
import random as random
from CEO.cards.eventlistener import EventListenerInterface, PrintAllEventListener
import CEO.cards.deck as deck
from CEO.cards.hand import *
import CEO.cards.round as rd
from CEO.cards.simplebehavior import SimpleBehaviorBase
import CEO.cards.player as player
from gym_ceo.envs.features import WillWinTrick_AfterState
from gym_ceo.envs.seat_ceo_features_env import (
    SeatCEOFeaturesEnv,
    TriplesUnderValueCount,
    ValuesInRangeCount,
    OtherPlayerHandCount,
    HandCardCountRelative,
    HighestCard,
    SinglesUnderValueCountRelative,
    DoublesUnderValueCountRelative,
    TriplesUnderValueCountRelative,
    WillWinTrick_AfterState,
    HandSummary,
)
from gym_ceo.envs.seat_ceo_env import SeatCEOEnv
from gym_ceo.envs.observation import Observation, ObservationFactory
from stable_baselines3.common.env_checker import check_env

from gym_ceo.envs.actions import Actions, ActionEnum, ActionSpaceFactory

import numpy as np


class MockPlayerBehavior(player.PlayerBehaviorInterface, SimpleBehaviorBase):
    value_to_play: list[CardValue]
    to_play_next_index: int

    def __init__(self):
        self.value_to_play = []
        self.to_play_next_index = 0

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


def create_ceo_env(
    hands: list[Hand], num_players=4, action_space_type="ceo"
) -> tuple[SeatCEOEnv, Observation]:
    # Make the players
    behaviors = []
    for i in range(num_players - 1):
        behaviors.append(MockPlayerBehavior())

    env = SeatCEOEnv(
        action_space_type=action_space_type,
        num_players=num_players,
        behaviors=behaviors,
        hands=hands,
        listener=PrintAllEventListener(),
        skip_passing=True,
    )
    factory = ObservationFactory(env.num_players)

    observation_array = env.reset()
    observation = factory.create_observation(array=observation_array)

    return env, observation


def test_SeatCEOFeaturesEnv_check_env():
    """
    Test SeatCEOFeaturesEnv using the Gym check_env
    """

    listener = EventListenerInterface()
    listener = PrintAllEventListener()

    print("Checking SeatCEOFeaturesEnv. Seed 0")
    random.seed(0)
    full_env = SeatCEOEnv(listener=listener)
    env = SeatCEOFeaturesEnv(full_env)
    check_env(env, True, True)

    print("Checking SeatCEOFeaturesEnv. Seed 1")
    random.seed(1)
    full_env = SeatCEOEnv(listener=listener)
    env = SeatCEOFeaturesEnv(full_env)
    check_env(env, True, True)

    print("Checking SeatCEOFeaturesEnv. Seed 2")
    random.seed(2)
    full_env = SeatCEOEnv(listener=listener)
    env = SeatCEOFeaturesEnv(full_env)
    check_env(env, True, True)


def test_TriplesUnderValueCount():
    """
    Test the TriplesUnderValueCount feature
    """

    # Create CardValue objects for ease of use later
    cv0 = CardValue(0)
    cv1 = CardValue(1)
    cv2 = CardValue(2)
    cv3 = CardValue(3)
    cv4 = CardValue(4)
    cv5 = CardValue(5)
    cv6 = CardValue(5)
    cv7 = CardValue(5)
    cv8 = CardValue(5)
    cv9 = CardValue(5)
    cv10 = CardValue(10)
    cv11 = CardValue(11)
    cv12 = CardValue(12)

    # Make the hands
    hand1 = Hand()
    hand1.add_cards(cv0, 1)
    hand1.add_cards(cv3, 3)
    hand1.add_cards(cv11, 3)
    hand1.add_cards(cv12, 3)

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
    )
    factory = ObservationFactory(env.num_players)

    feature_calc = TriplesUnderValueCount(env, threshold=10)

    observation_array = env.reset()
    observation = factory.create_observation(array=observation_array)

    feature_array = np.zeros(1)
    info = dict()
    feature_calc.calc(observation, feature_array, 0, info)

    assert feature_array[0] == 1


def test_SinglesUnderValueCountRelative():
    """
    Test the SinglesUnderValueCountRelative feature
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
    cv9 = CardValue(9)
    cv10 = CardValue(10)
    cv11 = CardValue(11)
    cv12 = CardValue(12)

    # Make the other players' hands
    hand2 = Hand()
    hand2.add_cards(cv1, 1)

    hand3 = Hand()
    hand3.add_cards(cv2, 1)

    hand4 = Hand()
    hand4.add_cards(cv3, 1)

    # Test where the highest card is an ace
    hand1 = Hand()
    hand1.add_cards(cv0, 2)
    hand1.add_cards(cv2, 1)
    hand1.add_cards(cv3, 1)
    hand1.add_cards(cv4, 2)
    hand1.add_cards(cv5, 1)
    hand1.add_cards(cv6, 1)
    hand1.add_cards(cv11, 1)
    hand1.add_cards(cv12, 1)

    hands = [hand1, hand2, hand3, hand4]

    env, observation = create_ceo_env(hands)

    feature_5 = SinglesUnderValueCountRelative(env, relative_threshold=5, max_value=5)
    feature_6 = SinglesUnderValueCountRelative(env, relative_threshold=6, max_value=5)
    feature_7 = SinglesUnderValueCountRelative(env, relative_threshold=7, max_value=5)
    feature_8 = SinglesUnderValueCountRelative(env, relative_threshold=8, max_value=5)

    feature_array = np.zeros(1)
    info = dict()

    feature_5.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 4

    feature_6.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 3

    feature_7.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 2

    feature_8.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 2

    # Test where the highest card is a king
    hand1 = Hand()
    hand1.add_cards(cv0, 2)
    hand1.add_cards(cv2, 1)
    hand1.add_cards(cv3, 1)
    hand1.add_cards(cv4, 2)
    hand1.add_cards(cv5, 1)
    hand1.add_cards(cv6, 1)
    hand1.add_cards(cv11, 1)

    hands = [hand1, hand2, hand3, hand4]

    env, observation = create_ceo_env(hands)

    feature_5 = SinglesUnderValueCountRelative(env, relative_threshold=5, max_value=5)
    feature_6 = SinglesUnderValueCountRelative(env, relative_threshold=6, max_value=5)
    feature_7 = SinglesUnderValueCountRelative(env, relative_threshold=7, max_value=5)
    feature_8 = SinglesUnderValueCountRelative(env, relative_threshold=8, max_value=5)

    feature_array = np.zeros(1)
    info = dict()

    feature_5.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 3

    feature_6.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 2

    feature_7.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 2

    feature_8.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 1

    # Test where the highest card is a six
    hand1 = Hand()
    hand1.add_cards(cv0, 1)
    hand1.add_cards(cv2, 1)
    hand1.add_cards(cv3, 1)
    hand1.add_cards(cv4, 2)
    hand1.add_cards(cv5, 1)
    hand1.add_cards(cv6, 1)

    hands = [hand1, hand2, hand3, hand4]

    env, observation = create_ceo_env(hands)

    feature_5 = SinglesUnderValueCountRelative(env, relative_threshold=5, max_value=5)
    feature_6 = SinglesUnderValueCountRelative(env, relative_threshold=6, max_value=5)
    feature_7 = SinglesUnderValueCountRelative(env, relative_threshold=7, max_value=5)
    feature_8 = SinglesUnderValueCountRelative(env, relative_threshold=8, max_value=5)

    feature_array = np.zeros(1)
    info = dict()

    feature_5.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 1

    feature_6.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 0

    feature_7.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 0

    feature_8.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 0

    # Test max_value
    hand1 = Hand()
    hand1.add_cards(cv0, 1)
    hand1.add_cards(cv2, 1)
    hand1.add_cards(cv3, 1)
    hand1.add_cards(cv4, 1)
    hand1.add_cards(cv5, 2)
    hand1.add_cards(cv6, 2)

    hands = [hand1, hand2, hand3, hand4]

    env, observation = create_ceo_env(hands)

    feature_max2 = SinglesUnderValueCountRelative(env, relative_threshold=1, max_value=2)
    feature_max3 = SinglesUnderValueCountRelative(env, relative_threshold=1, max_value=3)
    feature_max4 = SinglesUnderValueCountRelative(env, relative_threshold=1, max_value=4)
    feature_max5 = SinglesUnderValueCountRelative(env, relative_threshold=1, max_value=5)

    assert feature_max2.max_value == 2
    assert feature_max3.max_value == 3
    assert feature_max4.max_value == 4
    assert feature_max5.max_value == 5

    feature_array = np.zeros(1)
    info = dict()

    feature_max2.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 2

    feature_max3.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 3

    feature_max4.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 4

    feature_max5.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 4


def test_DoublesUnderValueCountRelative():
    """
    Test the DoublesUnderValueCountRelative feature
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
    cv9 = CardValue(9)
    cv10 = CardValue(10)
    cv11 = CardValue(11)
    cv12 = CardValue(12)

    # Make the other players' hands
    hand2 = Hand()
    hand2.add_cards(cv1, 1)

    hand3 = Hand()
    hand3.add_cards(cv2, 1)

    hand4 = Hand()
    hand4.add_cards(cv3, 1)

    # Test where the highest card is an ace
    hand1 = Hand()
    hand1.add_cards(cv0, 1)
    hand1.add_cards(cv2, 2)
    hand1.add_cards(cv3, 2)
    hand1.add_cards(cv4, 1)
    hand1.add_cards(cv5, 2)
    hand1.add_cards(cv6, 2)
    hand1.add_cards(cv11, 2)
    hand1.add_cards(cv12, 2)

    hands = [hand1, hand2, hand3, hand4]

    env, observation = create_ceo_env(hands)

    feature_5 = DoublesUnderValueCountRelative(env, relative_threshold=5, max_value=5)
    feature_6 = DoublesUnderValueCountRelative(env, relative_threshold=6, max_value=5)
    feature_7 = DoublesUnderValueCountRelative(env, relative_threshold=7, max_value=5)
    feature_8 = DoublesUnderValueCountRelative(env, relative_threshold=8, max_value=5)

    feature_array = np.zeros(1)
    info = dict()

    feature_5.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 4

    feature_6.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 3

    feature_7.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 2

    feature_8.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 2

    # Test where the highest card is a king
    hand1 = Hand()
    hand1.add_cards(cv0, 1)
    hand1.add_cards(cv2, 2)
    hand1.add_cards(cv3, 2)
    hand1.add_cards(cv4, 1)
    hand1.add_cards(cv5, 2)
    hand1.add_cards(cv6, 2)
    hand1.add_cards(cv11, 2)

    hands = [hand1, hand2, hand3, hand4]

    env, observation = create_ceo_env(hands)

    feature_5 = DoublesUnderValueCountRelative(env, relative_threshold=5, max_value=5)
    feature_6 = DoublesUnderValueCountRelative(env, relative_threshold=6, max_value=5)
    feature_7 = DoublesUnderValueCountRelative(env, relative_threshold=7, max_value=5)
    feature_8 = DoublesUnderValueCountRelative(env, relative_threshold=8, max_value=5)

    feature_array = np.zeros(1)
    info = dict()

    feature_5.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 3

    feature_6.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 2

    feature_7.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 2

    feature_8.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 1

    # Test where the highest card is a six
    hand1 = Hand()
    hand1.add_cards(cv0, 2)
    hand1.add_cards(cv2, 2)
    hand1.add_cards(cv3, 2)
    hand1.add_cards(cv4, 1)
    hand1.add_cards(cv5, 2)
    hand1.add_cards(cv6, 2)

    hands = [hand1, hand2, hand3, hand4]

    env, observation = create_ceo_env(hands)

    feature_5 = DoublesUnderValueCountRelative(env, relative_threshold=5, max_value=5)
    feature_6 = DoublesUnderValueCountRelative(env, relative_threshold=6, max_value=5)
    feature_7 = DoublesUnderValueCountRelative(env, relative_threshold=7, max_value=5)
    feature_8 = DoublesUnderValueCountRelative(env, relative_threshold=8, max_value=5)

    feature_array = np.zeros(1)
    info = dict()

    feature_5.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 1

    feature_6.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 0

    feature_7.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 0

    feature_8.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 0

    # Test max_value
    hand1 = Hand()
    hand1.add_cards(cv0, 2)
    hand1.add_cards(cv2, 2)
    hand1.add_cards(cv3, 2)
    hand1.add_cards(cv4, 2)
    hand1.add_cards(cv5, 1)
    hand1.add_cards(cv6, 1)

    hands = [hand1, hand2, hand3, hand4]

    env, observation = create_ceo_env(hands)

    feature_max2 = DoublesUnderValueCountRelative(env, relative_threshold=1, max_value=2)
    feature_max3 = DoublesUnderValueCountRelative(env, relative_threshold=1, max_value=3)
    feature_max4 = DoublesUnderValueCountRelative(env, relative_threshold=1, max_value=4)
    feature_max5 = DoublesUnderValueCountRelative(env, relative_threshold=1, max_value=5)

    assert feature_max2.max_value == 2
    assert feature_max3.max_value == 3
    assert feature_max4.max_value == 4
    assert feature_max5.max_value == 5

    feature_array = np.zeros(1)
    info = dict()

    feature_max2.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 2

    feature_max3.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 3

    feature_max4.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 4

    feature_max5.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 4


def test_TriplesUnderValueCountRelative():
    """
    Test the TriplesUnderValueCountRelative feature
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
    cv9 = CardValue(9)
    cv10 = CardValue(10)
    cv11 = CardValue(11)
    cv12 = CardValue(12)

    # Make the other players' hands
    hand2 = Hand()
    hand2.add_cards(cv1, 1)

    hand3 = Hand()
    hand3.add_cards(cv2, 1)

    hand4 = Hand()
    hand4.add_cards(cv3, 1)

    # Test where the highest card is an ace
    hand1 = Hand()
    hand1.add_cards(cv0, 1)
    hand1.add_cards(cv2, 4)
    hand1.add_cards(cv3, 3)
    hand1.add_cards(cv4, 1)
    hand1.add_cards(cv5, 4)
    hand1.add_cards(cv6, 3)
    hand1.add_cards(cv11, 3)
    hand1.add_cards(cv12, 3)

    hands = [hand1, hand2, hand3, hand4]

    env, observation = create_ceo_env(hands)

    feature_5 = TriplesUnderValueCountRelative(env, relative_threshold=5, max_value=5)
    feature_6 = TriplesUnderValueCountRelative(env, relative_threshold=6, max_value=5)
    feature_7 = TriplesUnderValueCountRelative(env, relative_threshold=7, max_value=5)
    feature_8 = TriplesUnderValueCountRelative(env, relative_threshold=8, max_value=5)

    feature_array = np.zeros(1)
    info = dict()

    feature_5.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 4

    feature_6.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 3

    feature_7.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 2

    feature_8.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 2

    # Test where the highest card is a king
    hand1 = Hand()
    hand1.add_cards(cv0, 1)
    hand1.add_cards(cv2, 4)
    hand1.add_cards(cv3, 3)
    hand1.add_cards(cv4, 1)
    hand1.add_cards(cv5, 4)
    hand1.add_cards(cv6, 3)
    hand1.add_cards(cv11, 3)

    hands = [hand1, hand2, hand3, hand4]

    env, observation = create_ceo_env(hands)

    feature_5 = TriplesUnderValueCountRelative(env, relative_threshold=5, max_value=5)
    feature_6 = TriplesUnderValueCountRelative(env, relative_threshold=6, max_value=5)
    feature_7 = TriplesUnderValueCountRelative(env, relative_threshold=7, max_value=5)
    feature_8 = TriplesUnderValueCountRelative(env, relative_threshold=8, max_value=5)

    feature_array = np.zeros(1)
    info = dict()

    feature_5.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 3

    feature_6.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 2

    feature_7.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 2

    feature_8.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 1

    # Test where the highest card is a six
    hand1 = Hand()
    hand1.add_cards(cv0, 3)
    hand1.add_cards(cv2, 4)
    hand1.add_cards(cv3, 3)
    hand1.add_cards(cv4, 1)
    hand1.add_cards(cv5, 4)
    hand1.add_cards(cv6, 3)

    hands = [hand1, hand2, hand3, hand4]

    env, observation = create_ceo_env(hands)

    feature_5 = TriplesUnderValueCountRelative(env, relative_threshold=5, max_value=5)
    feature_6 = TriplesUnderValueCountRelative(env, relative_threshold=6, max_value=5)
    feature_7 = TriplesUnderValueCountRelative(env, relative_threshold=7, max_value=5)
    feature_8 = TriplesUnderValueCountRelative(env, relative_threshold=8, max_value=5)

    feature_array = np.zeros(1)
    info = dict()

    feature_5.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 1

    feature_6.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 0

    feature_7.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 0

    feature_8.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 0

    # Test max_value
    hand1 = Hand()
    hand1.add_cards(cv0, 3)
    hand1.add_cards(cv2, 4)
    hand1.add_cards(cv3, 3)
    hand1.add_cards(cv4, 4)
    hand1.add_cards(cv5, 1)
    hand1.add_cards(cv6, 1)

    hands = [hand1, hand2, hand3, hand4]

    env, observation = create_ceo_env(hands)

    feature_max2 = TriplesUnderValueCountRelative(env, relative_threshold=1, max_value=2)
    feature_max3 = TriplesUnderValueCountRelative(env, relative_threshold=1, max_value=3)
    feature_max4 = TriplesUnderValueCountRelative(env, relative_threshold=1, max_value=4)
    feature_max5 = TriplesUnderValueCountRelative(env, relative_threshold=1, max_value=5)

    assert feature_max2.max_value == 2
    assert feature_max3.max_value == 3
    assert feature_max4.max_value == 4
    assert feature_max5.max_value == 5

    feature_array = np.zeros(1)
    info = dict()

    feature_max2.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 2

    feature_max3.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 3

    feature_max4.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 4

    feature_max5.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 4


def test_ValuesInRangeCount():
    """
    Test the ValuesInRangeCount feature
    """

    # Create CardValue objects for ease of use later
    cv0 = CardValue(0)
    cv1 = CardValue(1)
    cv2 = CardValue(2)
    cv3 = CardValue(3)
    cv4 = CardValue(4)
    cv5 = CardValue(5)
    cv6 = CardValue(5)
    cv7 = CardValue(5)
    cv8 = CardValue(5)
    cv9 = CardValue(5)
    cv10 = CardValue(10)
    cv11 = CardValue(11)
    cv12 = CardValue(12)

    # Make the hands
    hand1 = Hand()
    hand1.add_cards(cv0, 1)
    hand1.add_cards(cv3, 3)
    hand1.add_cards(cv11, 3)
    hand1.add_cards(cv12, 3)

    hand2 = Hand()
    hand2.add_cards(cv1, 1)
    hand2.add_cards(cv2, 1)
    hand2.add_cards(cv4, 2)

    hand3 = Hand()
    hand3.add_cards(cv0, 1)
    hand3.add_cards(cv2, 1)
    hand3.add_cards(cv5, 2)

    hand4 = Hand()
    hand4.add_cards(cv1, 1)
    hand4.add_cards(cv2, 2)
    hand4.add_cards(cv3, 1)

    hands = [hand1, hand2, hand3, hand4]

    # Make the players
    behavior2 = MockPlayerBehavior()
    behavior3 = MockPlayerBehavior()
    behavior4 = MockPlayerBehavior()
    behaviors = [behavior2, behavior3, behavior4]

    env = SeatCEOEnv(
        num_players=4,
        behaviors=behaviors,
        hands=hands,
        listener=PrintAllEventListener(),
        skip_passing=True,
    )
    factory = ObservationFactory(env.num_players)

    feature_calc_0_3 = ValuesInRangeCount(env, range_begin=0, range_end=3, max_value=3)
    feature_calc_0_4 = ValuesInRangeCount(env, range_begin=0, range_end=4, max_value=3)
    feature_calc_3_6 = ValuesInRangeCount(env, range_begin=3, range_end=6, max_value=3)
    feature_calc_4_6 = ValuesInRangeCount(env, range_begin=4, range_end=6, max_value=3)

    observation_array = env.reset()
    observation = factory.create_observation(array=observation_array)

    feature_array = np.zeros(1)
    info = dict()

    feature_calc_0_3.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 1

    feature_calc_0_4.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 2

    feature_calc_3_6.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 1

    feature_calc_4_6.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 0

    # Test max_value
    feature_calc_max2 = ValuesInRangeCount(env, range_begin=0, range_end=12, max_value=2)
    feature_calc_max3 = ValuesInRangeCount(env, range_begin=0, range_end=12, max_value=3)

    feature_calc_max2.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 2

    feature_calc_max3.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 3


def test_OtherPlayerHandCount():
    """
    Test the OtherPlayerHandCount feature
    """

    # Create CardValue objects for ease of use later
    cv0 = CardValue(0)
    cv1 = CardValue(1)
    cv2 = CardValue(2)
    cv3 = CardValue(3)
    cv4 = CardValue(4)
    cv5 = CardValue(5)
    cv6 = CardValue(5)
    cv7 = CardValue(5)
    cv8 = CardValue(5)
    cv9 = CardValue(5)
    cv10 = CardValue(10)
    cv11 = CardValue(11)
    cv12 = CardValue(12)

    # Make the hands
    hand1 = Hand()
    hand1.add_cards(cv0, 1)
    hand1.add_cards(cv3, 3)
    hand1.add_cards(cv11, 3)
    hand1.add_cards(cv12, 3)

    hand2 = Hand()
    hand2.add_cards(cv1, 1)
    hand2.add_cards(cv2, 1)
    hand2.add_cards(cv4, 2)

    hand3 = Hand()
    hand3.add_cards(cv0, 1)
    hand3.add_cards(cv2, 1)
    hand3.add_cards(cv5, 3)

    hand4 = Hand()
    hand4.add_cards(cv1, 3)
    hand4.add_cards(cv2, 3)
    hand4.add_cards(cv3, 3)

    hands = [hand1, hand2, hand3, hand4]

    # Make the players
    behavior2 = MockPlayerBehavior()
    behavior3 = MockPlayerBehavior()
    behavior4 = MockPlayerBehavior()
    behaviors = [behavior2, behavior3, behavior4]

    env = SeatCEOEnv(
        num_players=4,
        behaviors=behaviors,
        hands=hands,
        listener=PrintAllEventListener(),
        skip_passing=True,
    )
    factory = ObservationFactory(env.num_players)

    feature_0 = OtherPlayerHandCount(env, other_player_index=0, max_value=5)
    feature_1 = OtherPlayerHandCount(env, other_player_index=1, max_value=5)
    feature_2 = OtherPlayerHandCount(env, other_player_index=2, max_value=5)

    observation_array = env.reset()
    observation = factory.create_observation(array=observation_array)

    feature_array = np.zeros(1)
    info = dict()

    feature_0.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 4

    feature_1.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 5

    feature_2.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 5

    # Test max_value
    feature_0 = OtherPlayerHandCount(env, other_player_index=0, max_value=4)
    feature_1 = OtherPlayerHandCount(env, other_player_index=1, max_value=4)
    feature_2 = OtherPlayerHandCount(env, other_player_index=2, max_value=4)

    observation_array = env.reset()
    observation = factory.create_observation(array=observation_array)

    feature_array = np.zeros(1)
    info = dict()

    feature_0.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 4

    feature_1.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 4

    feature_2.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 4


def test_HandCardCountRelative():
    """
    Test the HandCardCountRelative feature
    """

    # Create CardValue objects for ease of use later
    cv0 = CardValue(0)
    cv1 = CardValue(1)
    cv2 = CardValue(2)
    cv3 = CardValue(3)
    cv4 = CardValue(4)
    cv5 = CardValue(5)
    cv6 = CardValue(5)
    cv7 = CardValue(5)
    cv8 = CardValue(5)
    cv9 = CardValue(5)
    cv10 = CardValue(10)
    cv11 = CardValue(11)
    cv12 = CardValue(12)

    # Make non-ceo hands
    hand2 = Hand()
    hand2.add_cards(cv1, 1)
    hand2.add_cards(cv2, 1)
    hand2.add_cards(cv4, 2)

    hand3 = Hand()
    hand3.add_cards(cv0, 1)
    hand3.add_cards(cv2, 1)
    hand3.add_cards(cv5, 3)

    hand4 = Hand()
    hand4.add_cards(cv1, 3)
    hand4.add_cards(cv2, 3)
    hand4.add_cards(cv3, 3)

    # Test when the highest card is an ace
    hand1 = Hand()
    hand1.add_cards(cv0, 1)
    hand1.add_cards(cv3, 3)
    hand1.add_cards(cv11, 2)
    hand1.add_cards(cv12, 1)

    hands = [hand1, hand2, hand3, hand4]

    env, observation = create_ceo_env(hands)

    feature_0 = HandCardCountRelative(env, relative_card_value=0, max_value=4)
    feature_1 = HandCardCountRelative(env, relative_card_value=-1, max_value=4)
    feature_2 = HandCardCountRelative(env, relative_card_value=-2, max_value=4)

    feature_array = np.zeros(1)
    info = dict()

    feature_0.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 1

    feature_1.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 2

    feature_2.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 0

    # Test when the highest card is a king
    hand1 = Hand()
    hand1.add_cards(cv0, 1)
    hand1.add_cards(cv3, 3)
    hand1.add_cards(cv10, 2)
    hand1.add_cards(cv11, 1)

    hands = [hand1, hand2, hand3, hand4]

    env, observation = create_ceo_env(hands)

    feature_0 = HandCardCountRelative(env, relative_card_value=0, max_value=4)
    feature_1 = HandCardCountRelative(env, relative_card_value=-1, max_value=4)
    feature_2 = HandCardCountRelative(env, relative_card_value=-2, max_value=4)

    feature_array = np.zeros(1)
    info = dict()

    feature_0.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 1

    feature_1.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 2

    feature_2.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 0

    # Test small max value
    hand1 = Hand()
    hand1.add_cards(cv0, 2)

    hands = [hand1, hand2, hand3, hand4]

    env, observation = create_ceo_env(hands)

    feature_0 = HandCardCountRelative(env, relative_card_value=0, max_value=4)
    feature_1 = HandCardCountRelative(env, relative_card_value=-1, max_value=4)
    feature_2 = HandCardCountRelative(env, relative_card_value=-2, max_value=4)

    feature_array = np.zeros(1)
    info = dict()

    feature_0.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 2

    feature_1.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 0

    feature_2.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 0

    # Test max value
    hand1 = Hand()
    hand1.add_cards(cv0, 1)
    hand1.add_cards(cv3, 3)
    hand1.add_cards(cv10, 8)
    hand1.add_cards(cv12, 4)

    hands = [hand1, hand2, hand3, hand4]

    env, observation = create_ceo_env(hands)

    feature_0 = HandCardCountRelative(env, relative_card_value=0, max_value=3)
    feature_1 = HandCardCountRelative(env, relative_card_value=-1, max_value=3)
    feature_2 = HandCardCountRelative(env, relative_card_value=-2, max_value=3)

    feature_array = np.zeros(1)
    info = dict()

    feature_0.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 3

    feature_1.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 0

    feature_2.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 3


def test_HighestCard():
    """
    Test the HighestCard feature
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
    cv9 = CardValue(9)
    cv10 = CardValue(10)
    cv11 = CardValue(11)
    cv12 = CardValue(12)

    # Make non-ceo hands
    hand2 = Hand()
    hand2.add_cards(cv1, 1)
    hand2.add_cards(cv2, 1)
    hand2.add_cards(cv4, 2)

    hand3 = Hand()
    hand3.add_cards(cv0, 1)
    hand3.add_cards(cv2, 1)
    hand3.add_cards(cv5, 3)

    hand4 = Hand()
    hand4.add_cards(cv1, 3)
    hand4.add_cards(cv2, 3)
    hand4.add_cards(cv3, 3)

    # Test when the highest card is an ace
    hand1 = Hand()
    hand1.add_cards(cv0, 1)
    hand1.add_cards(cv3, 3)
    hand1.add_cards(cv11, 2)
    hand1.add_cards(cv12, 1)

    hands = [hand1, hand2, hand3, hand4]

    env, observation = create_ceo_env(hands)

    feature_6 = HighestCard(env, min_card_value=6)
    feature_10 = HighestCard(env, min_card_value=10)

    feature_array = np.zeros(1)
    info = dict()

    feature_6.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 6

    feature_10.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 2

    # Test when the highest card is a king
    hand1 = Hand()
    hand1.add_cards(cv0, 1)
    hand1.add_cards(cv3, 3)
    hand1.add_cards(cv11, 2)

    hands = [hand1, hand2, hand3, hand4]

    env, observation = create_ceo_env(hands)

    feature_6 = HighestCard(env, min_card_value=6)
    feature_10 = HighestCard(env, min_card_value=10)

    assert feature_6.max_value == 6
    assert feature_10.max_value == 2

    feature_array = np.zeros(1)
    info = dict()

    feature_6.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 5

    feature_10.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 1

    # Test when the highest card is an eight
    hand1 = Hand()
    hand1.add_cards(cv0, 1)
    hand1.add_cards(cv3, 3)
    hand1.add_cards(cv8, 2)

    hands = [hand1, hand2, hand3, hand4]

    env, observation = create_ceo_env(hands)

    feature_6 = HighestCard(env, min_card_value=6)
    feature_10 = HighestCard(env, min_card_value=10)

    feature_array = np.zeros(1)
    info = dict()

    feature_6.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 2

    feature_10.calc(observation, feature_array, 0, info)
    assert feature_array[0] == 0


def test_AfterState_WillWinTrick_AfterState():
    """Test the WillWinTrick feature with afterstates."""

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
    cv9 = CardValue(9)
    cv10 = CardValue(10)
    cv11 = CardValue(11)
    cv12 = CardValue(12)

    # Setup the environment
    hand2 = Hand()
    hand2.add_cards(cv4, 2)

    hand3 = Hand()
    hand3.add_cards(cv4, 3)

    hand4 = Hand()
    hand4.add_cards(cv4, 4)

    hand1 = Hand()
    hand1.add_cards(cv0, 5)
    hand1.add_cards(cv1, 1)
    hand1.add_cards(cv12, 1)

    hands = [hand1, hand2, hand3, hand4]

    env, observation = create_ceo_env(hands)
    observation_factory = env.observation_factory

    feature = WillWinTrick_AfterState(env)
    feature.notify_other_features([feature])

    feature_array = np.zeros(1)
    info = dict()

    # Set up for tests where the agent leads
    hands = [hand1, hand2, hand3, hand4]
    state = rd.RoundState(hands, None)

    observation = observation_factory.create_observation(
        type="lead", cur_hand=hand1, starting_player=0, state=state
    )

    # Test when the agent leads an ace
    action = env.action_space.find_full_action(ActionEnum.PLAY_HIGHEST_NUM)
    afterstate_array = env.get_afterstate(observation.get_array(), action)
    afterstate = observation_factory.create_observation(array=afterstate_array)

    feature.calc(afterstate, feature_array, 0, info)
    assert feature_array[0] == 1

    # Test when the agent leads a low card
    action = env.action_space.find_full_action(ActionEnum.PLAY_LOWEST_WITHOUT_BREAK_NUM)
    afterstate_array = env.get_afterstate(observation.get_array(), action)
    afterstate = observation_factory.create_observation(array=afterstate_array)

    feature.calc(afterstate, feature_array, 0, info)
    assert feature_array[0] == 0

    # Set up for tests where the agent plays last on a trick
    hands = [hand1, hand2, hand3, hand4]
    state = rd.RoundState(hands, 1)

    env.action_space = env._action_space_factory.create_play(hand1, cv0, 1)

    observation = observation_factory.create_observation(
        type="play",
        cur_hand=hand1,
        starting_player=1,
        cur_card_value=cv0,
        cur_card_count=1,
        state=state,
    )

    # Test when the agent plays
    action = env.action_space.find_full_action(ActionEnum.PLAY_HIGHEST_NUM)
    afterstate_array = env.get_afterstate(observation.get_array(), action)
    afterstate = observation_factory.create_observation(array=afterstate_array)

    feature.calc(afterstate, feature_array, 0, info)
    assert feature_array[0] == 1

    # Test when the agent passes
    action = env.action_space.find_full_action(ActionEnum.PASS_ON_TRICK_NUM)
    afterstate_array = env.get_afterstate(observation.get_array(), action)
    afterstate = observation_factory.create_observation(array=afterstate_array)

    feature.calc(afterstate, feature_array, 0, info)
    assert feature_array[0] == 0


def test_AfterState_WillWinTrick_AfterState_SixPlayers():
    """Test the WillWinTrick feature with afterstates. Here we have six players.
    Test when the feature value depends on the number of cards other players have."""

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
    cv9 = CardValue(9)
    cv10 = CardValue(10)
    cv11 = CardValue(11)
    cv12 = CardValue(12)

    # Setup the environment
    hand1 = Hand()
    hand1.add_cards(cv0, 1)
    hand1.add_cards(cv1, 5)
    hand1.add_cards(cv2, 5)
    hand1.add_cards(cv3, 5)
    hand1.add_cards(cv12, 5)

    hand2 = Hand()
    hand2.add_cards(cv4, 1)

    hand3 = Hand()
    hand3.add_cards(cv4, 2)

    hand4 = Hand()
    hand4.add_cards(cv4, 3)

    hand5 = Hand()
    hand5.add_cards(cv4, 4)

    hand6 = Hand()
    hand6.add_cards(cv4, 5)

    hands = [hand1, hand2, hand3, hand4, hand5, hand6]

    env, observation = create_ceo_env(hands, num_players=6, action_space_type="card")
    observation_factory = env.observation_factory

    feature_no_downstream = WillWinTrick_AfterState(env)
    feature_one_downstream = WillWinTrick_AfterState(env)
    feature_two_downstream = WillWinTrick_AfterState(env)

    feature_other_1 = OtherPlayerHandCount(env, other_player_index=1, max_value=5)
    feature_other_2 = OtherPlayerHandCount(env, other_player_index=2, max_value=5)

    feature_no_downstream.notify_other_features([feature_no_downstream])
    feature_one_downstream.notify_other_features([feature_one_downstream, feature_other_1])
    feature_two_downstream.notify_other_features(
        [feature_two_downstream, feature_other_1, feature_other_2]
    )

    feature_array = np.zeros(1)
    info = dict()

    # Set up for tests where the agent leads
    state = rd.RoundState(hands, None)

    observation = observation_factory.create_observation(
        type="lead", cur_hand=hand1, starting_player=0, state=state
    )

    # Test when the agent leads one card.
    afterstate_array = env.get_afterstate(observation.get_array(), 1)
    afterstate = observation_factory.create_observation(array=afterstate_array)

    feature_no_downstream.calc(afterstate, feature_array, 0, info)
    assert feature_array[0] == 0

    feature_one_downstream.calc(afterstate, feature_array, 0, info)
    assert feature_array[0] == 0

    feature_two_downstream.calc(afterstate, feature_array, 0, info)
    assert feature_array[0] == 0

    # Test where the agent plays last on a trick with one card
    starting_player = 1
    card_count = 1
    state = rd.RoundState(hands, starting_player)

    env.action_space = env._action_space_factory.create_play(hand1, cv0, 1)

    observation = observation_factory.create_observation(
        type="play",
        cur_hand=hand1,
        starting_player=starting_player,
        cur_card_value=cv0,
        cur_card_count=card_count,
        state=state,
    )

    afterstate_array = env.get_afterstate(observation.get_array(), 1)
    afterstate = observation_factory.create_observation(array=afterstate_array)

    feature_no_downstream.calc(afterstate, feature_array, 0, info)
    assert feature_array[0] == 1

    feature_one_downstream.calc(afterstate, feature_array, 0, info)
    assert feature_array[0] == 1

    feature_two_downstream.calc(afterstate, feature_array, 0, info)
    assert feature_array[0] == 1

    # Test where there is one more player to play on a trick with one card
    starting_player = 2
    card_count = 1
    state = rd.RoundState(hands, starting_player)

    env.action_space = env._action_space_factory.create_play(hand1, cv0, 1)

    observation = observation_factory.create_observation(
        type="play",
        cur_hand=hand1,
        starting_player=starting_player,
        cur_card_value=cv0,
        cur_card_count=card_count,
        state=state,
    )

    afterstate_array = env.get_afterstate(observation.get_array(), 1)
    afterstate = observation_factory.create_observation(array=afterstate_array)

    feature_no_downstream.calc(afterstate, feature_array, 0, info)
    assert feature_array[0] == 0

    feature_one_downstream.calc(afterstate, feature_array, 0, info)
    assert feature_array[0] == 0

    feature_two_downstream.calc(afterstate, feature_array, 0, info)
    assert feature_array[0] == 0

    # Test where there is one more player to play on a trick with two cards
    starting_player = 2
    card_count = 2
    state = rd.RoundState(hands, starting_player)

    env.action_space = env._action_space_factory.create_play(hand1, cv0, 1)

    observation = observation_factory.create_observation(
        type="play",
        cur_hand=hand1,
        starting_player=starting_player,
        cur_card_value=cv0,
        cur_card_count=card_count,
        state=state,
    )

    afterstate_array = env.get_afterstate(observation.get_array(), 1)
    afterstate = observation_factory.create_observation(array=afterstate_array)

    feature_no_downstream.calc(afterstate, feature_array, 0, info)
    assert feature_array[0] == 0

    feature_one_downstream.calc(afterstate, feature_array, 0, info)
    assert feature_array[0] == 1

    feature_two_downstream.calc(afterstate, feature_array, 0, info)
    assert feature_array[0] == 1

    # Test where there is one more player to play on a trick with three cards
    starting_player = 2
    card_count = 3
    state = rd.RoundState(hands, starting_player)

    env.action_space = env._action_space_factory.create_play(hand1, cv0, 1)

    observation = observation_factory.create_observation(
        type="play",
        cur_hand=hand1,
        starting_player=starting_player,
        cur_card_value=cv0,
        cur_card_count=card_count,
        state=state,
    )

    afterstate_array = env.get_afterstate(observation.get_array(), 1)
    afterstate = observation_factory.create_observation(array=afterstate_array)

    feature_no_downstream.calc(afterstate, feature_array, 0, info)
    assert feature_array[0] == 0

    feature_one_downstream.calc(afterstate, feature_array, 0, info)
    assert feature_array[0] == 1

    feature_two_downstream.calc(afterstate, feature_array, 0, info)
    assert feature_array[0] == 1

    # Test where there are two more players to play on a trick with one card
    starting_player = 3
    card_count = 1
    state = rd.RoundState(hands, starting_player)

    env.action_space = env._action_space_factory.create_play(hand1, cv0, 1)

    observation = observation_factory.create_observation(
        type="play",
        cur_hand=hand1,
        starting_player=starting_player,
        cur_card_value=cv0,
        cur_card_count=card_count,
        state=state,
    )

    afterstate_array = env.get_afterstate(observation.get_array(), 1)
    afterstate = observation_factory.create_observation(array=afterstate_array)

    feature_no_downstream.calc(afterstate, feature_array, 0, info)
    assert feature_array[0] == 0

    feature_one_downstream.calc(afterstate, feature_array, 0, info)
    assert feature_array[0] == 0

    feature_two_downstream.calc(afterstate, feature_array, 0, info)
    assert feature_array[0] == 0

    # Test where there are two more players to play on a trick with two cards
    starting_player = 3
    card_count = 2
    state = rd.RoundState(hands, starting_player)

    env.action_space = env._action_space_factory.create_play(hand1, cv0, 1)

    observation = observation_factory.create_observation(
        type="play",
        cur_hand=hand1,
        starting_player=starting_player,
        cur_card_value=cv0,
        cur_card_count=card_count,
        state=state,
    )

    afterstate_array = env.get_afterstate(observation.get_array(), 1)
    afterstate = observation_factory.create_observation(array=afterstate_array)

    feature_no_downstream.calc(afterstate, feature_array, 0, info)
    assert feature_array[0] == 0

    feature_one_downstream.calc(afterstate, feature_array, 0, info)
    assert feature_array[0] == 0

    feature_two_downstream.calc(afterstate, feature_array, 0, info)
    assert feature_array[0] == 0

    # Test where there are two more players to play on a trick with three cards
    starting_player = 3
    card_count = 3
    state = rd.RoundState(hands, starting_player)

    env.action_space = env._action_space_factory.create_play(hand1, cv0, 1)

    observation = observation_factory.create_observation(
        type="play",
        cur_hand=hand1,
        starting_player=starting_player,
        cur_card_value=cv0,
        cur_card_count=card_count,
        state=state,
    )

    afterstate_array = env.get_afterstate(observation.get_array(), 1)
    afterstate = observation_factory.create_observation(array=afterstate_array)

    feature_no_downstream.calc(afterstate, feature_array, 0, info)
    assert feature_array[0] == 0

    feature_one_downstream.calc(afterstate, feature_array, 0, info)
    assert feature_array[0] == 0

    feature_two_downstream.calc(afterstate, feature_array, 0, info)
    assert feature_array[0] == 1

    # Test where there are three more players to play on a trick with one card
    starting_player = 4
    card_count = 1
    state = rd.RoundState(hands, starting_player)

    env.action_space = env._action_space_factory.create_play(hand1, cv0, 1)

    observation = observation_factory.create_observation(
        type="play",
        cur_hand=hand1,
        starting_player=starting_player,
        cur_card_value=cv0,
        cur_card_count=card_count,
        state=state,
    )

    afterstate_array = env.get_afterstate(observation.get_array(), 1)
    afterstate = observation_factory.create_observation(array=afterstate_array)

    feature_no_downstream.calc(afterstate, feature_array, 0, info)
    assert feature_array[0] == 0

    feature_one_downstream.calc(afterstate, feature_array, 0, info)
    assert feature_array[0] == 0

    feature_two_downstream.calc(afterstate, feature_array, 0, info)
    assert feature_array[0] == 0

    # Test where the agent passes in a situation where he would win the trick if he played.
    starting_player = 2
    card_count = 2
    state = rd.RoundState(hands, starting_player)

    env.action_space = env._action_space_factory.create_play(hand1, cv0, 1)

    observation = observation_factory.create_observation(
        type="play",
        cur_hand=hand1,
        starting_player=starting_player,
        cur_card_value=cv0,
        cur_card_count=card_count,
        state=state,
    )

    action = env.action_space.n - 1
    afterstate_array = env.get_afterstate(observation.get_array(), action)
    afterstate = observation_factory.create_observation(array=afterstate_array)

    feature_no_downstream.calc(afterstate, feature_array, 0, info)
    assert feature_array[0] == 0

    feature_one_downstream.calc(afterstate, feature_array, 0, info)
    assert feature_array[0] == 0

    feature_two_downstream.calc(afterstate, feature_array, 0, info)
    assert feature_array[0] == 0


def test_HandSummary():
    """
    Test the HandSummary features
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
    cv9 = CardValue(9)
    cv10 = CardValue(10)
    cv11 = CardValue(11)
    cv12 = CardValue(12)

    # Setup the environment
    hand1 = Hand()
    hand1.add_cards(cv0, 1)

    hand2 = Hand()
    hand2.add_cards(cv4, 1)

    hand3 = Hand()
    hand3.add_cards(cv4, 2)

    hand4 = Hand()
    hand4.add_cards(cv4, 3)

    hands = [hand1, hand2, hand3, hand4]

    env, observation = create_ceo_env(hands, num_players=4)
    observation_factory = env.observation_factory

    feature_2_3 = HandSummary(
        env,
        high_card_exact_count=2,
        high_card_obs_max=13,
        bucket_count=3,
        bucket_obs_max=13,
        include_hand_count=True,
    )

    feature_3_2 = HandSummary(
        env,
        high_card_exact_count=3,
        high_card_obs_max=13,
        bucket_count=2,
        bucket_obs_max=13,
        include_hand_count=True,
    )

    feature_array = np.zeros(100)
    info = dict()

    # Test when there is one of each card
    hand1 = Hand()
    hand1.add_cards(cv0, 1)
    hand1.add_cards(cv1, 1)
    hand1.add_cards(cv2, 1)
    hand1.add_cards(cv3, 1)
    hand1.add_cards(cv4, 1)
    hand1.add_cards(cv5, 1)
    hand1.add_cards(cv6, 1)
    hand1.add_cards(cv7, 1)
    hand1.add_cards(cv8, 1)
    hand1.add_cards(cv9, 1)
    hand1.add_cards(cv10, 1)
    hand1.add_cards(cv11, 1)
    hand1.add_cards(cv12, 1)

    hands = [hand1, hand2, hand3, hand4]
    state = rd.RoundState(hands, None)

    observation = observation_factory.create_observation(
        type="lead", cur_hand=hand1, starting_player=0, state=state
    )

    i = 0

    feature_2_3.calc(observation, feature_array, i, info)
    assert feature_2_3.get_high_card(feature_array, i) == 12
    assert feature_2_3.get_high_card_count(feature_array, i, 0) == 1
    assert feature_2_3.get_high_card_count(feature_array, i, 1) == 1
    assert feature_2_3.get_bucket_card_count(feature_array, i, 0) == 4
    assert feature_2_3.get_bucket_card_count(feature_array, i, 1) == 4
    assert feature_2_3.get_bucket_card_count(feature_array, i, 2) == 3
    assert feature_2_3.get_hand_card_count(feature_array, i) == 13

    feature_3_2.calc(observation, feature_array, i, info)
    assert feature_3_2.get_high_card(feature_array, i) == 12
    assert feature_3_2.get_high_card_count(feature_array, i, 0) == 1
    assert feature_3_2.get_high_card_count(feature_array, i, 1) == 1
    assert feature_3_2.get_high_card_count(feature_array, i, 2) == 1
    assert feature_3_2.get_bucket_card_count(feature_array, i, 0) == 5
    assert feature_3_2.get_bucket_card_count(feature_array, i, 1) == 5
    assert feature_3_2.get_hand_card_count(feature_array, i) == 13

    # Test when there is one of each card except the ace
    hand1 = Hand()
    hand1.add_cards(cv0, 1)
    hand1.add_cards(cv1, 1)
    hand1.add_cards(cv2, 1)
    hand1.add_cards(cv3, 1)
    hand1.add_cards(cv4, 1)
    hand1.add_cards(cv5, 1)
    hand1.add_cards(cv6, 1)
    hand1.add_cards(cv7, 1)
    hand1.add_cards(cv8, 1)
    hand1.add_cards(cv9, 1)
    hand1.add_cards(cv10, 1)
    hand1.add_cards(cv11, 1)
    hand1.add_cards(cv12, 0)

    hands = [hand1, hand2, hand3, hand4]
    state = rd.RoundState(hands, None)

    observation = observation_factory.create_observation(
        type="lead", cur_hand=hand1, starting_player=0, state=state
    )

    i = 0

    feature_2_3.calc(observation, feature_array, i, info)
    assert feature_2_3.get_high_card(feature_array, i) == 11
    assert feature_2_3.get_high_card_count(feature_array, i, 0) == 1
    assert feature_2_3.get_high_card_count(feature_array, i, 1) == 1
    assert feature_2_3.get_bucket_card_count(feature_array, i, 0) == 4
    assert feature_2_3.get_bucket_card_count(feature_array, i, 1) == 3
    assert feature_2_3.get_bucket_card_count(feature_array, i, 2) == 3
    assert feature_2_3.get_hand_card_count(feature_array, i) == 12

    feature_3_2.calc(observation, feature_array, i, info)
    assert feature_3_2.get_high_card(feature_array, i) == 11
    assert feature_3_2.get_high_card_count(feature_array, i, 0) == 1
    assert feature_3_2.get_high_card_count(feature_array, i, 1) == 1
    assert feature_3_2.get_high_card_count(feature_array, i, 2) == 1
    assert feature_3_2.get_bucket_card_count(feature_array, i, 0) == 5
    assert feature_3_2.get_bucket_card_count(feature_array, i, 1) == 4
    assert feature_3_2.get_hand_card_count(feature_array, i) == 12

    # Test when there is one of each card except the ace and king
    hand1 = Hand()
    hand1.add_cards(cv0, 1)
    hand1.add_cards(cv1, 1)
    hand1.add_cards(cv2, 1)
    hand1.add_cards(cv3, 1)
    hand1.add_cards(cv4, 1)
    hand1.add_cards(cv5, 1)
    hand1.add_cards(cv6, 1)
    hand1.add_cards(cv7, 1)
    hand1.add_cards(cv8, 1)
    hand1.add_cards(cv9, 1)
    hand1.add_cards(cv10, 1)
    hand1.add_cards(cv11, 0)
    hand1.add_cards(cv12, 0)

    hands = [hand1, hand2, hand3, hand4]
    state = rd.RoundState(hands, None)

    observation = observation_factory.create_observation(
        type="lead", cur_hand=hand1, starting_player=0, state=state
    )

    i = 0

    feature_2_3.calc(observation, feature_array, i, info)
    assert feature_2_3.get_high_card(feature_array, i) == 10
    assert feature_2_3.get_high_card_count(feature_array, i, 0) == 1
    assert feature_2_3.get_high_card_count(feature_array, i, 1) == 1
    assert feature_2_3.get_bucket_card_count(feature_array, i, 0) == 3
    assert feature_2_3.get_bucket_card_count(feature_array, i, 1) == 3
    assert feature_2_3.get_bucket_card_count(feature_array, i, 2) == 3
    assert feature_2_3.get_hand_card_count(feature_array, i) == 11

    feature_3_2.calc(observation, feature_array, i, info)
    assert feature_3_2.get_high_card(feature_array, i) == 10
    assert feature_3_2.get_high_card_count(feature_array, i, 0) == 1
    assert feature_3_2.get_high_card_count(feature_array, i, 1) == 1
    assert feature_3_2.get_high_card_count(feature_array, i, 2) == 1
    assert feature_3_2.get_bucket_card_count(feature_array, i, 0) == 4
    assert feature_3_2.get_bucket_card_count(feature_array, i, 1) == 4
    assert feature_3_2.get_hand_card_count(feature_array, i) == 11

    # Test when there are different numbers of high cards
    hand1 = Hand()
    hand1.add_cards(cv0, 0)
    hand1.add_cards(cv1, 0)
    hand1.add_cards(cv2, 0)
    hand1.add_cards(cv3, 0)
    hand1.add_cards(cv4, 0)
    hand1.add_cards(cv5, 0)
    hand1.add_cards(cv6, 0)
    hand1.add_cards(cv7, 0)
    hand1.add_cards(cv8, 0)
    hand1.add_cards(cv9, 0)
    hand1.add_cards(cv10, 1)
    hand1.add_cards(cv11, 2)
    hand1.add_cards(cv12, 3)

    hands = [hand1, hand2, hand3, hand4]
    state = rd.RoundState(hands, None)

    observation = observation_factory.create_observation(
        type="lead", cur_hand=hand1, starting_player=0, state=state
    )

    feature_2_3.calc(observation, feature_array, i, info)
    assert feature_2_3.get_high_card(feature_array, i) == 12
    assert feature_2_3.get_high_card_count(feature_array, i, 0) == 3
    assert feature_2_3.get_high_card_count(feature_array, i, 1) == 2
    assert feature_2_3.get_bucket_card_count(feature_array, i, 0) == 0
    assert feature_2_3.get_bucket_card_count(feature_array, i, 1) == 0
    assert feature_2_3.get_bucket_card_count(feature_array, i, 2) == 1
    assert feature_2_3.get_hand_card_count(feature_array, i) == 6

    feature_3_2.calc(observation, feature_array, i, info)
    assert feature_3_2.get_high_card(feature_array, i) == 12
    assert feature_3_2.get_high_card_count(feature_array, i, 0) == 3
    assert feature_3_2.get_high_card_count(feature_array, i, 1) == 2
    assert feature_3_2.get_high_card_count(feature_array, i, 2) == 1
    assert feature_3_2.get_bucket_card_count(feature_array, i, 0) == 0
    assert feature_3_2.get_bucket_card_count(feature_array, i, 1) == 0
    assert feature_3_2.get_hand_card_count(feature_array, i) == 6

    # Test when there are exactly enough low cards to fill the exact card counts and
    # the buckets.
    hand1 = Hand()
    hand1.add_cards(cv0, 1)
    hand1.add_cards(cv1, 2)
    hand1.add_cards(cv2, 3)
    hand1.add_cards(cv3, 4)
    hand1.add_cards(cv4, 5)
    hand1.add_cards(cv5, 0)
    hand1.add_cards(cv6, 0)
    hand1.add_cards(cv7, 0)
    hand1.add_cards(cv8, 0)
    hand1.add_cards(cv9, 0)
    hand1.add_cards(cv10, 0)
    hand1.add_cards(cv11, 0)
    hand1.add_cards(cv12, 0)

    hands = [hand1, hand2, hand3, hand4]
    state = rd.RoundState(hands, None)

    observation = observation_factory.create_observation(
        type="lead", cur_hand=hand1, starting_player=0, state=state
    )

    feature_2_3.calc(observation, feature_array, i, info)
    assert feature_2_3.get_high_card(feature_array, i) == 4
    assert feature_2_3.get_high_card_count(feature_array, i, 0) == 5
    assert feature_2_3.get_high_card_count(feature_array, i, 1) == 4
    assert feature_2_3.get_bucket_card_count(feature_array, i, 0) == 1
    assert feature_2_3.get_bucket_card_count(feature_array, i, 1) == 2
    assert feature_2_3.get_bucket_card_count(feature_array, i, 2) == 3
    assert feature_2_3.get_hand_card_count(feature_array, i) == 15

    feature_3_2.calc(observation, feature_array, i, info)
    assert feature_3_2.get_high_card(feature_array, i) == 4
    assert feature_3_2.get_high_card_count(feature_array, i, 0) == 5
    assert feature_3_2.get_high_card_count(feature_array, i, 1) == 4
    assert feature_3_2.get_high_card_count(feature_array, i, 2) == 3
    assert feature_3_2.get_bucket_card_count(feature_array, i, 0) == 1
    assert feature_3_2.get_bucket_card_count(feature_array, i, 1) == 2
    assert feature_3_2.get_hand_card_count(feature_array, i) == 15

    # Test when there are not enough low cards to fill the exact card counts and
    # the buckets.
    hand1 = Hand()
    hand1.add_cards(cv0, 1)
    hand1.add_cards(cv1, 2)
    hand1.add_cards(cv2, 3)
    hand1.add_cards(cv3, 4)
    hand1.add_cards(cv4, 0)
    hand1.add_cards(cv5, 0)
    hand1.add_cards(cv6, 0)
    hand1.add_cards(cv7, 0)
    hand1.add_cards(cv8, 0)
    hand1.add_cards(cv9, 0)
    hand1.add_cards(cv10, 0)
    hand1.add_cards(cv11, 0)
    hand1.add_cards(cv12, 0)

    hands = [hand1, hand2, hand3, hand4]
    state = rd.RoundState(hands, None)

    observation = observation_factory.create_observation(
        type="lead", cur_hand=hand1, starting_player=0, state=state
    )

    feature_2_3.calc(observation, feature_array, i, info)
    assert feature_2_3.get_high_card(feature_array, i) == 4
    assert feature_2_3.get_high_card_count(feature_array, i, 0) == 0
    assert feature_2_3.get_high_card_count(feature_array, i, 1) == 4
    assert feature_2_3.get_bucket_card_count(feature_array, i, 0) == 1
    assert feature_2_3.get_bucket_card_count(feature_array, i, 1) == 2
    assert feature_2_3.get_bucket_card_count(feature_array, i, 2) == 3
    assert feature_2_3.get_hand_card_count(feature_array, i) == 10

    feature_3_2.calc(observation, feature_array, i, info)
    assert feature_3_2.get_high_card(feature_array, i) == 4
    assert feature_3_2.get_high_card_count(feature_array, i, 0) == 0
    assert feature_3_2.get_high_card_count(feature_array, i, 1) == 4
    assert feature_3_2.get_high_card_count(feature_array, i, 2) == 3
    assert feature_3_2.get_bucket_card_count(feature_array, i, 0) == 1
    assert feature_3_2.get_bucket_card_count(feature_array, i, 1) == 2
    assert feature_3_2.get_hand_card_count(feature_array, i) == 10
