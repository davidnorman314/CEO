"""Tests for CEOAnySeatEnv where the agent's seat changes each episode."""

import random

from stable_baselines3.common.env_checker import check_env

import ceo.game.player as player
import ceo.game.round as rd
from ceo.envs.ceo_any_seat_env import CEOAnySeatEnv
from ceo.envs.observation import Observation
from ceo.game.eventlistener import EventListenerInterface
from ceo.game.hand import CardValue, Hand
from ceo.game.simplebehavior import SimpleBehaviorBase


class MockPlayerBehavior(player.PlayerBehaviorInterface, SimpleBehaviorBase):
    """Mock behavior that plays predetermined cards."""

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
            raise AssertionError("No more values to play")

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
            raise AssertionError("No more values to play")

        ret = self.value_to_play[self.to_play_next_index]
        self.to_play_next_index += 1
        return ret


def create_play(hand: Hand, behavior: MockPlayerBehavior, cv: CardValue, count: int):
    """Add cards to hand and schedule behavior to play them."""
    hand.add_cards(cv, count)
    behavior.value_to_play.append(cv)


def test_ceo_any_seat_env_creation():
    """Test that CEOAnySeatEnv can be created and includes seat in observation."""
    env = CEOAnySeatEnv(num_players=6)
    assert env.num_players == 6
    assert env.observation_factory._include_seat_number is True


def test_ceo_any_seat_env_observation_includes_seat():
    """Test that observations include the seat number."""
    env = CEOAnySeatEnv(
        num_players=6,
        action_space_type="all_card",
    )

    obs_array, info = env.reset(seed=42)

    obs = Observation(env.observation_factory, array=obs_array)
    seat = obs.get_seat_number()

    assert seat is not None
    assert 0 <= seat < 6
    assert seat == env._current_seat


def test_ceo_any_seat_env_seat_changes_between_resets():
    """Test that the seat changes between resets."""
    env = CEOAnySeatEnv(
        num_players=6,
        action_space_type="all_card",
    )

    seats_seen = set()

    for i in range(50):
        env.reset(seed=i)
        seats_seen.add(env._current_seat)

    # Should have seen multiple different seats
    assert len(seats_seen) > 1


def test_ceo_any_seat_env_seeded_reset_reproducible():
    """Test that seeded resets produce the same seat."""
    env1 = CEOAnySeatEnv(num_players=6, action_space_type="all_card")
    env2 = CEOAnySeatEnv(num_players=6, action_space_type="all_card")

    env1.reset(seed=12345)
    env2.reset(seed=12345)

    assert env1._current_seat == env2._current_seat


def test_ceo_any_seat_env_info_contains_seat():
    """Test that the info dict at end of episode contains seat."""
    cv0 = CardValue(0)
    cv1 = CardValue(1)
    cv2 = CardValue(2)
    cv3 = CardValue(3)

    hand0 = Hand()
    hand1 = Hand()
    hand2 = Hand()
    hand3 = Hand()

    behavior0 = MockPlayerBehavior()
    behavior1 = MockPlayerBehavior()
    behavior2 = MockPlayerBehavior()
    behavior3 = MockPlayerBehavior()

    # Simple game: each player has one card
    create_play(hand0, behavior0, cv0, 1)
    create_play(hand1, behavior1, cv1, 1)
    create_play(hand2, behavior2, cv2, 1)
    create_play(hand3, behavior3, cv3, 1)

    hands = [hand0, hand1, hand2, hand3]
    behaviors = [behavior0, behavior1, behavior2, behavior3]

    env = CEOAnySeatEnv(
        num_players=4,
        behaviors=behaviors,
        hands=hands,
        skip_passing=True,
        action_space_type="all_card",
    )

    obs, info = env.reset(seed=0)
    seat = env._current_seat

    # Step until done
    done = False
    final_info = None
    while not done:
        obs_obj = Observation(env.observation_factory, array=obs)
        action = None
        for cv in range(13):
            if obs_obj.get_card_count(cv) > 0:
                action = cv
                break

        if action is None:
            action = 13

        obs, reward, done, truncated, final_info = env.step(action)

    assert "seat" in final_info
    assert final_info["seat"] == seat


def test_ceo_any_seat_env_check_env():
    """Test using Gym's check_env with all_card action space."""
    listener = EventListenerInterface()

    random.seed(0)
    env = CEOAnySeatEnv(
        num_players=6,
        listener=listener,
        action_space_type="all_card",
    )
    check_env(env, warn=True, skip_render_check=True)


def test_ceo_any_seat_env_check_env_4_players():
    """Test check_env with 4 players."""
    listener = EventListenerInterface()

    random.seed(42)
    env = CEOAnySeatEnv(
        num_players=4,
        listener=listener,
        action_space_type="all_card",
    )
    check_env(env, warn=True, skip_render_check=True)


def test_ceo_any_seat_env_complete_episode():
    """Test a complete game episode runs to completion."""
    env = CEOAnySeatEnv(
        num_players=4,
        action_space_type="all_card",
        obs_kwargs={"include_valid_actions": True},
    )

    obs, info = env.reset(seed=42)

    obs_obj = Observation(env.observation_factory, array=obs)
    seat = obs_obj.get_seat_number()
    assert seat is not None
    assert seat == env._current_seat

    # Play until done
    done = False
    steps = 0
    max_steps = 200

    while not done and steps < max_steps:
        obs_obj = Observation(env.observation_factory, array=obs)

        action = None
        for cv in range(13):
            if obs_obj.get_play_card_action_valid(cv) > 0:
                action = cv
                break

        if action is None:
            action = 13

        obs, reward, done, truncated, info = env.step(action)
        steps += 1

    assert done or truncated, "Episode should complete or truncate"
    if done and not truncated:
        assert "seat" in info
        assert info["seat"] == seat


def test_ceo_any_seat_env_all_seats_reachable():
    """Test that all seats are reachable through random selection."""
    env = CEOAnySeatEnv(num_players=4, action_space_type="all_card")

    seats_seen = set()

    for i in range(200):
        env.reset(seed=i * 7)
        seats_seen.add(env._current_seat)

        if len(seats_seen) == 4:
            break

    assert seats_seen == {0, 1, 2, 3}


def test_ceo_any_seat_env_observation_factory_updates_per_episode():
    """Test that observation factory seat is updated each episode."""
    env = CEOAnySeatEnv(num_players=6, action_space_type="all_card")

    for i in range(20):
        env.reset(seed=i * 13)
        # Factory seat should always match current seat
        assert env.observation_factory._seat_number == env._current_seat
