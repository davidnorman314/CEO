import random as random

import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

import CEO.cards.player as player
import CEO.cards.round as rd
from CEO.cards.eventlistener import PrintAllEventListener
from CEO.cards.hand import *
from CEO.cards.simplebehavior import SimpleBehaviorBase
from gym_ceo.envs.actions import Actions
from gym_ceo.envs.observation import Observation, ObservationFactory
from gym_ceo.envs.observation_hand import ObservationHand
from gym_ceo.envs.seat_ceo_env import SeatCEOEnv


class MockPlayerBehavior(player.PlayerBehaviorInterface, SimpleBehaviorBase):
    def __init__(self):
        pass

    def pass_cards(self, hand: Hand, count: int) -> list[CardValue]:
        pass

    def lead(self, player_position: int, hand: Hand, state) -> CardValue:
        pass

    def play_on_trick(
        self,
        starting_position: int,
        player_position: int,
        hand: Hand,
        cur_trick_value: CardValue,
        cur_trick_count: int,
        state: rd.RoundState,
    ) -> CardValue:
        pass


def create_ceo_env(hand1: Hand) -> tuple[SeatCEOEnv, Observation]:
    # Make the other players' hands
    cv0 = CardValue(0)
    cv1 = CardValue(1)
    cv2 = CardValue(2)

    hand2 = Hand()
    hand2.add_cards(cv0, 1)

    hand3 = Hand()
    hand3.add_cards(cv1, 1)

    hand4 = Hand()
    hand4.add_cards(cv2, 1)

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
    factory = ObservationFactory(env.num_players, env.seat_number)

    observation_array, info = env.reset()
    observation = factory.create_observation(array=observation_array)

    return env, observation


def test_ObservationHand():
    """
    Test ObservationHand
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

    # Test where the highest card is an ace
    hand1 = Hand()
    hand1.add_cards(cv0, 2)
    hand1.add_cards(cv2, 1)
    hand1.add_cards(cv3, 3)
    hand1.add_cards(cv4, 2)
    hand1.add_cards(cv5, 1)
    hand1.add_cards(cv6, 1)
    hand1.add_cards(cv11, 1)
    hand1.add_cards(cv12, 1)

    env, observation = create_ceo_env(hand1)
    observation_hand = ObservationHand(observation)

    # Test getting the total number of cards
    assert observation_hand.card_count() == hand1.card_count()

    # Test getting the card counts
    for cv_num in range(0, 13):
        cv = CardValue(cv_num)

        assert observation_hand.count(cv) == hand1.count(cv)

    # Test getting the maximum value
    assert observation_hand.max_card_value() == hand1.max_card_value()

    # Test playing cards
    observation_hand.play_cards(PlayedCards(cv3, 2))
    hand1.play_cards(PlayedCards(cv3, 2))

    for cv_num in range(0, 13):
        cv = CardValue(cv_num)

        assert observation_hand.count(cv) == hand1.count(cv)

    # Test playing the highest card
    observation_hand.play_cards(PlayedCards(cv12, 1))
    hand1.play_cards(PlayedCards(cv12, 1))

    assert observation_hand.card_count() == hand1.card_count()
    assert observation_hand.max_card_value() == hand1.max_card_value()
    for cv_num in range(0, 13):
        cv = CardValue(cv_num)

        assert observation_hand.count(cv) == hand1.count(cv)

    # Test playing the lowest card
    observation_hand.play_cards(PlayedCards(cv0, 2))
    hand1.play_cards(PlayedCards(cv0, 2))

    assert observation_hand.card_count() == hand1.card_count()
    assert observation_hand.max_card_value() == hand1.max_card_value()
    for cv_num in range(0, 13):
        cv = CardValue(cv_num)

        assert observation_hand.count(cv) == hand1.count(cv)
