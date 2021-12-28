import pytest
import random as random
from CEO.cards.eventlistener import EventListenerInterface, PrintAllEventListener
import CEO.cards.deck as deck
from CEO.cards.hand import *
import CEO.cards.round as rd
from CEO.cards.simplebehavior import SimpleBehaviorBase
import CEO.cards.player as player
from gym_ceo.envs.seat_ceo_features_env import (
    SeatCEOFeaturesEnv,
    TriplesUnderValueCount,
)
from gym_ceo.envs.seat_ceo_env import SeatCEOEnv
from gym_ceo.envs.observation import Observation, ObservationFactory
from stable_baselines3.common.env_checker import check_env

from gym_ceo.envs.actions import Actions

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

    feature_calc = TriplesUnderValueCount(env, 10)

    observation_array = env.reset()
    observation = factory.create_observation(array=observation_array)

    feature_array = np.zeros(1)
    info = dict()
    feature_calc.calc(observation, feature_array, 0, info)

    assert feature_array[0] == 1
