"""Tests for the seat number inclusion feature in ObservationFactory and Observation."""

import numpy as np
import torch as th

from ceo.envs.observation import Observation, ObservationFactory
from ceo.game.hand import CardValue, Hand


class MockRoundState:
    """Mock RoundState for testing observations."""

    def __init__(self, num_players: int, cards_remaining: list[int]):
        self.cards_remaining = cards_remaining
        self.last_player_to_play_index = 0


def test_include_seat_number_increases_dimension_by_one():
    """Test that including seat number adds exactly one dimension."""
    for num_players in [4, 5, 6]:
        factory_without = ObservationFactory(num_players=num_players, seat_number=0)
        factory_with = ObservationFactory(
            num_players=num_players, seat_number=0, include_seat_number=True
        )

        assert (
            factory_with.observation_dimension
            == factory_without.observation_dimension + 1
        )

        # Also check with valid actions
        factory_without_va = ObservationFactory(
            num_players=num_players, seat_number=0, include_valid_actions=True
        )
        factory_with_va = ObservationFactory(
            num_players=num_players,
            seat_number=0,
            include_seat_number=True,
            include_valid_actions=True,
        )

        assert (
            factory_with_va.observation_dimension
            == factory_without_va.observation_dimension + 1
        )


def test_lead_observation_with_seat_number():
    """Test complete lead observation with seat number included."""
    num_players = 4
    seat = 2

    factory = ObservationFactory(
        num_players=num_players, seat_number=seat, include_seat_number=True
    )

    hand = Hand()
    hand.add_cards(CardValue(0), 2)
    hand.add_cards(CardValue(5), 3)
    hand.add_cards(CardValue(12), 1)

    # Cards remaining: seat 0 has 10, seat 1 has 8, seat 2 (agent) has 6, seat 3 has 5
    cards_remaining = [10, 8, 6, 5]
    state = MockRoundState(num_players, cards_remaining)

    obs = Observation(
        factory,
        type="lead",
        starting_player=seat,
        cur_hand=hand,
        state=state,
    )

    # Seat number
    assert obs.get_seat_number() == seat

    # Hand cards
    assert obs.get_card_count(0) == 2
    assert obs.get_card_count(5) == 3
    assert obs.get_card_count(12) == 1
    assert obs.get_card_count(3) == 0

    # Other players (excluding seat 2): [0, 1, 3] in order
    assert obs.get_other_player_card_count(0) == 10  # seat 0
    assert obs.get_other_player_card_count(1) == 8   # seat 1
    assert obs.get_other_player_card_count(2) == 5   # seat 3

    # Trick state for lead
    assert obs.get_cur_trick_value() is None
    assert obs.get_cur_trick_count() == 0


def test_play_observation_with_seat_number():
    """Test complete play observation with seat number included."""
    num_players = 6
    seat = 3

    factory = ObservationFactory(
        num_players=num_players, seat_number=seat, include_seat_number=True
    )

    hand = Hand()
    hand.add_cards(CardValue(5), 3)
    hand.add_cards(CardValue(8), 2)

    cards_remaining = [10, 9, 8, 5, 7, 6]
    state = MockRoundState(num_players, cards_remaining)
    state.last_player_to_play_index = 1

    obs = Observation(
        factory,
        type="play",
        starting_player=0,
        cur_index=seat,
        cur_hand=hand,
        cur_card_value=CardValue(4),
        cur_card_count=2,
        state=state,
    )

    # Seat number
    assert obs.get_seat_number() == seat

    # Hand cards
    assert obs.get_card_count(5) == 3
    assert obs.get_card_count(8) == 2
    assert obs.get_card_count(0) == 0

    # Other players (excluding seat 3): [0, 1, 2, 4, 5] in order
    assert obs.get_other_player_card_count(0) == 10  # seat 0
    assert obs.get_other_player_card_count(1) == 9   # seat 1
    assert obs.get_other_player_card_count(2) == 8   # seat 2
    assert obs.get_other_player_card_count(3) == 7   # seat 4
    assert obs.get_other_player_card_count(4) == 6   # seat 5

    # Trick state
    assert obs.get_cur_trick_value() == 4
    assert obs.get_cur_trick_count() == 2
    assert obs.get_starting_player() == 0
    assert obs.get_last_player() == 1


def test_seat_number_works_for_all_seats():
    """Test that seat number is correctly stored for all possible seats."""
    num_players = 6

    for seat in range(num_players):
        factory = ObservationFactory(
            num_players=num_players, seat_number=seat, include_seat_number=True
        )

        hand = Hand()
        hand.add_cards(CardValue(5), 3)

        state = MockRoundState(num_players, [10] * num_players)

        obs = Observation(
            factory,
            type="lead",
            starting_player=seat,
            cur_hand=hand,
            state=state,
        )

        assert obs.get_seat_number() == seat
        # Verify hand still works
        assert obs.get_card_count(5) == 3


def test_get_seat_number_returns_none_when_not_included():
    """Test that get_seat_number returns None when seat not in observation."""
    factory = ObservationFactory(num_players=6, seat_number=2)

    hand = Hand()
    hand.add_cards(CardValue(5), 3)

    state = MockRoundState(6, [3, 10, 10, 10, 10, 10])

    obs = Observation(
        factory,
        type="lead",
        starting_player=2,
        cur_hand=hand,
        state=state,
    )

    assert obs.get_seat_number() is None
    # Other getters still work
    assert obs.get_card_count(5) == 3


def test_observation_from_array_with_seat():
    """Test creating observation from array preserves seat and other values."""
    factory = ObservationFactory(
        num_players=6, seat_number=3, include_seat_number=True
    )

    # Create an array with known values
    array = np.zeros(factory.observation_dimension)
    array[0] = 3  # seat number
    array[1] = 2  # 2 cards of value 0
    array[6] = 3  # 3 cards of value 5

    obs = Observation(factory, array=array)

    assert obs.get_seat_number() == 3
    assert obs.get_card_count(0) == 2
    assert obs.get_card_count(5) == 3


def test_observation_from_tensor_with_seat():
    """Test creating observation from PyTorch tensor preserves seat."""
    factory = ObservationFactory(
        num_players=6, seat_number=4, include_seat_number=True
    )

    tensor = th.zeros(factory.observation_dimension)
    tensor[0] = 4  # seat number
    tensor[1] = 1  # 1 card of value 0

    obs = Observation(factory, tensor=tensor)

    assert obs.get_seat_number() == 4
    assert obs.get_card_count(0) == 1


def test_observation_copy_preserves_seat():
    """Test that copying an observation preserves all values including seat."""
    factory = ObservationFactory(
        num_players=6, seat_number=5, include_seat_number=True
    )

    hand = Hand()
    hand.add_cards(CardValue(5), 3)

    state = MockRoundState(6, [3, 10, 10, 10, 10, 10])

    obs = Observation(
        factory,
        type="lead",
        starting_player=5,
        cur_hand=hand,
        state=state,
    )

    obs_copy = obs.copy()

    assert obs_copy.get_seat_number() == 5
    assert obs_copy.get_card_count(5) == 3
    assert obs_copy.get_other_player_card_count(0) == 3
