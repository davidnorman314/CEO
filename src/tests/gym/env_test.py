import random as random

import pytest
from stable_baselines3.common.env_checker import check_env

import CEO.cards.player as player
import CEO.cards.round as rd
from CEO.cards.eventlistener import EventListenerInterface, PrintAllEventListener
from CEO.cards.hand import CardValue, Hand
from CEO.cards.simplebehavior import SimpleBehaviorBase
from gym_ceo.envs.actions import ActionEnum, ActionSpaceFactory
from gym_ceo.envs.observation import Observation
from gym_ceo.envs.seat_ceo_env import SeatCEOEnv


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


def test_seatceoenv_check_env():
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


def test_seatceoenv_all_card_action_space_check_env():
    """
    Test using the Gym check_env were the environment uses the AllCardActionSpace.
    """

    listener = EventListenerInterface()
    listener = PrintAllEventListener()

    print("Checking SeatCEOEnv all_card. Seed 0")
    random.seed(0)
    env = SeatCEOEnv(listener=listener, action_space_type="all_card")
    check_env(env, True, True)

    print("Checking SeatCEOEnv all_card. Seed 1")
    random.seed(1)
    env = SeatCEOEnv(listener=listener)
    check_env(env, True, True)

    print("Checking SeatCEOEnv all_card. Seed 2")
    random.seed(2)
    env = SeatCEOEnv(listener=listener)
    check_env(env, True, True)


def test_seatceoenv_only_play_pass():
    """
    Test when the the agent's only valid play is to pass. The agent should
    not need to perform the action.
    """

    # Create CardValue objects for ease of use later
    cv0 = CardValue(0)
    cv1 = CardValue(1)
    cv2 = CardValue(2)
    cv3 = CardValue(3)
    cv4 = CardValue(4)
    cv5 = CardValue(5)
    cv6 = CardValue(6)

    # Make the hands. Note that we disable passing below
    hand1 = Hand()
    hand1.add_cards(cv0, 1)
    hand1.add_cards(cv3, 2)
    hand1.add_cards(cv2, 1)

    hand2 = Hand()
    hand2.add_cards(cv1, 1)
    hand2.add_cards(cv4, 2)
    hand2.add_cards(cv3, 1)
    hand2.add_cards(cv3, 3)

    hand3 = Hand()
    hand3.add_cards(cv2, 1)
    hand3.add_cards(cv5, 2)
    hand3.add_cards(cv0, 1)
    hand3.add_cards(cv4, 3)

    hand4 = Hand()
    hand4.add_cards(cv3, 1)
    hand4.add_cards(cv6, 5)
    hand4.add_cards(cv2, 2)
    hand4.add_cards(cv1, 1)
    hand4.add_cards(cv5, 3)

    hands = [hand1, hand2, hand3, hand4]

    # Make the players
    behavior2 = MockPlayerBehavior()
    behavior3 = MockPlayerBehavior()
    behavior4 = MockPlayerBehavior()

    # action: Play cv0
    behavior2.value_to_play.append(cv1)
    behavior3.value_to_play.append(cv2)
    behavior4.value_to_play.append(cv3)

    behavior4.value_to_play.append(cv6)
    behavior2.value_to_play.append(None)
    behavior3.value_to_play.append(None)

    behavior4.value_to_play.append(cv2)
    # action: Play cv3
    behavior2.value_to_play.append(cv4)
    behavior3.value_to_play.append(cv5)

    behavior3.value_to_play.append(cv0)
    behavior4.value_to_play.append(cv1)
    # action: Play cv2
    behavior2.value_to_play.append(cv3)

    behavior2.value_to_play.append(cv3)
    behavior3.value_to_play.append(cv4)
    behavior4.value_to_play.append(cv5)

    behaviors = [behavior2, behavior3, behavior4]

    obs_kwargs = {"include_valid_actions": True}

    env = SeatCEOEnv(
        num_players=4,
        behaviors=behaviors,
        hands=hands,
        listener=PrintAllEventListener(),
        skip_passing=True,
        action_space_type="all_card",
        obs_kwargs=obs_kwargs,
    )

    observation_array, _ = env.reset()
    observation = Observation(env.observation_factory, array=observation_array)
    assert observation.get_cur_trick_value() is None

    # Lead
    action = 0
    observation_array, reward, done, _, info = env.step(action)

    assert not done
    assert reward == 0.0

    observation = Observation(env.observation_factory, array=observation_array)
    assert observation.get_cur_trick_value() == 2

    # Play cv3
    action = 3
    observation_array, reward, done, _, info = env.step(action)

    assert not done
    assert reward == 0

    observation = Observation(env.observation_factory, array=observation_array)
    assert observation.get_cur_trick_value() == 1

    # Play cv2
    action = 2
    observation_array, reward, done, _, info = env.step(action)

    assert done
    assert reward == 1.0


def test_seatceoenv_all_card_action_space_negative_reward_lead():
    """
    Test when the action space returns a negative reward and ends the episode
    for invalid actions. Here the invalid action is passing when the agent
    needs to lead.
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

    # Make the hands. Note that we disable passing below
    hand1 = Hand()
    hand1.add_cards(cv0, 1)
    hand1.add_cards(cv3, 2)
    hand1.add_cards(cv7, 2)

    hand2 = Hand()
    hand2.add_cards(cv1, 1)
    hand2.add_cards(cv4, 2)
    hand2.add_cards(cv2, 1)
    hand2.add_cards(cv6, 2)

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

    # action: Pass, which isn't valid.
    behavior2.value_to_play.append(None)
    behavior3.value_to_play.append(None)
    behavior4.value_to_play.append(None)

    # action: Lead lowest
    behavior2.value_to_play.append(cv1)
    behavior3.value_to_play.append(cv2)
    behavior4.value_to_play.append(cv3)

    behavior4.value_to_play.append(cv2)
    # action: Play invalid card, which gets clipped to passing.
    behavior2.value_to_play.append(cv4)
    behavior3.value_to_play.append(cv5)

    behavior3.value_to_play.append(cv0)
    behavior4.value_to_play.append(cv1)
    behavior2.value_to_play.append(cv2)

    behavior2.value_to_play.append(cv6)

    behaviors = [behavior2, behavior3, behavior4]

    env = SeatCEOEnv(
        num_players=4,
        behaviors=behaviors,
        hands=hands,
        listener=PrintAllEventListener(),
        skip_passing=True,
        action_space_type="all_card",
    )

    observation = env.reset()

    # Do the pass action when we should lead.
    action = 13
    observation, reward, done, _truncated, info = env.step(action)

    assert done
    remaining_cards = 5.0
    assert reward == pytest.approx(-(2.0 + 8.0 * remaining_cards / 13.0))


def test_seatceoenv_all_card_action_space_negative_reward_invalid_card():
    """
    Test when the action space returns a negative reward and ends the episode
    for invalid actions. Here the agent's action is to lead an invalid card.
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

    # Make the hands. Note that we disable passing below
    hand1 = Hand()
    hand1.add_cards(cv0, 1)
    hand1.add_cards(cv3, 2)
    hand1.add_cards(cv7, 2)

    hand2 = Hand()
    hand2.add_cards(cv1, 1)
    hand2.add_cards(cv4, 2)
    hand2.add_cards(cv2, 1)
    hand2.add_cards(cv6, 2)

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

    # action: Lead highest
    # No other players can play.
    behavior2.value_to_play.append(None)
    behavior3.value_to_play.append(None)
    behavior4.value_to_play.append(None)

    # action: Lead lowest
    behavior2.value_to_play.append(cv1)
    behavior3.value_to_play.append(cv2)
    behavior4.value_to_play.append(cv3)

    behavior4.value_to_play.append(cv2)
    # action: Play invalid card, which gets clipped to passing.
    behavior2.value_to_play.append(cv4)
    behavior3.value_to_play.append(cv5)

    behavior3.value_to_play.append(cv0)
    behavior4.value_to_play.append(cv1)
    behavior2.value_to_play.append(cv2)

    behavior2.value_to_play.append(cv6)

    behaviors = [behavior2, behavior3, behavior4]

    env = SeatCEOEnv(
        num_players=4,
        behaviors=behaviors,
        hands=hands,
        listener=PrintAllEventListener(),
        skip_passing=True,
        action_space_type="all_card",
    )

    observation = env.reset()

    # Lead highest
    action = 7
    observation, reward, done, _truncated, info = env.step(action)

    assert not done
    assert reward == 0.0

    # Lead lowest
    action = 0
    observation, reward, done, _truncated, info = env.step(action)

    assert not done
    assert reward == 0

    # Lead a card that isn't in the hand.
    action = 12
    observation, reward, done, _truncated, info = env.step(action)

    remaining_cards = 2.0
    assert reward == pytest.approx(-(2.0 + 8.0 * remaining_cards / 13.0))
    assert done


def test_seatceoenv_all_card_action_space_action_order_matches_observation_order():
    """
    Test that the order of actions in the AllCardActionSpace match the order of
    actions in the observation when the observation includes which actions are valid.
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

    # Make the hands. Note that we disable passing below
    hand1 = Hand()
    hand1.add_cards(cv0, 1)
    hand1.add_cards(cv3, 2)
    hand1.add_cards(cv7, 2)

    hand2 = Hand()
    hand2.add_cards(cv1, 1)
    hand2.add_cards(cv4, 2)
    hand2.add_cards(cv2, 1)
    hand2.add_cards(cv6, 2)

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

    # action: Lead highest
    # No other players can play.
    behavior2.value_to_play.append(None)
    behavior3.value_to_play.append(None)
    behavior4.value_to_play.append(None)

    # action: Lead lowest
    behavior2.value_to_play.append(cv1)
    behavior3.value_to_play.append(cv2)
    behavior4.value_to_play.append(cv3)

    behavior4.value_to_play.append(cv2)
    # action: Play invalid card, which gets clipped to passing.
    behavior2.value_to_play.append(cv4)
    behavior3.value_to_play.append(cv5)

    behavior3.value_to_play.append(cv0)
    behavior4.value_to_play.append(cv1)
    behavior2.value_to_play.append(cv2)

    behavior2.value_to_play.append(cv6)

    behaviors = [behavior2, behavior3, behavior4]

    obs_kwargs = {"include_valid_actions": True}

    env = SeatCEOEnv(
        num_players=4,
        behaviors=behaviors,
        hands=hands,
        listener=PrintAllEventListener(),
        skip_passing=True,
        action_space_type="all_card",
        obs_kwargs=obs_kwargs,
    )

    action_space = env.action_space

    observation_array, _ = env.reset()
    observation = Observation(env.observation_factory, array=observation_array)
    valid_play_array = observation.get_valid_action_array()

    assert valid_play_array[action_space._pass_action] == 0.0
    assert valid_play_array[0] == 1.0
    assert valid_play_array[1] == 0.0
    assert valid_play_array[2] == 0.0
    assert valid_play_array[3] == 1.0
    assert valid_play_array[4] == 0.0
    assert valid_play_array[7] == 1.0

    # Lead highest
    action = 7
    observation_array, reward, done, _truncated, info = env.step(action)

    assert not done
    assert reward == 0.0

    observation = Observation(env.observation_factory, array=observation_array)
    valid_play_array = observation.get_valid_action_array()

    assert valid_play_array[action_space._pass_action] == 0.0
    assert valid_play_array[0] == 1.0
    assert valid_play_array[1] == 0.0
    assert valid_play_array[2] == 0.0
    assert valid_play_array[3] == 1.0
    assert valid_play_array[4] == 0.0
    assert valid_play_array[7] == 0.0

    # Lead lowest
    action = 0
    observation_array, reward, done, _truncated, info = env.step(action)

    assert not done
    assert reward == 0

    observation = Observation(env.observation_factory, array=observation_array)

    assert observation.get_cur_trick_value() == 2
    assert observation.get_cur_trick_value() == 2
    assert observation.get_last_player() == 3

    valid_play_array = observation.get_valid_action_array()

    assert valid_play_array[action_space._pass_action] == 1.0
    assert valid_play_array[0] == 0.0
    assert valid_play_array[1] == 0.0
    assert valid_play_array[2] == 0.0
    assert valid_play_array[3] == 1.0
    assert valid_play_array[4] == 0.0
    assert valid_play_array[7] == 0.0


def test_seatceoenv_passing():
    """
    Test the environment that models a player in the the CEO seat. Here we test that
    passing cards at the beginning of the round happens. For other tests, see below
    where passing is disabled.
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

    observation_array, _ = env.reset()
    observation = Observation(env.observation_factory, array=observation_array)

    assert observation.get_card_count(0) == 0
    assert observation.get_card_count(1) == 0
    assert observation.get_card_count(7) == 2


def test_seatceoenv_no_passing():
    """
    Test the environment that models a player in the the CEO seat. Here we disable
    passing to make the tests easier.
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

    assert env.action_space == ActionSpaceFactory.action_space_two_legal_lead

    # Lead lowest
    action = env.action_space.find_full_action(ActionEnum.PLAY_LOWEST_WITHOUT_BREAK_NUM)
    observation, reward, done, _truncated, info = env.step(action)

    assert env.action_space == ActionSpaceFactory.action_space_one_legal_play
    assert not done
    assert reward == 0

    action = env.action_space.find_full_action(ActionEnum.PLAY_HIGHEST_NUM)
    observation, reward, done, _truncated, info = env.step(action)

    assert reward > 0
    assert done


def test_seatceoenv_ceo_leads_and_no_one_plays():
    """
    Test the environment that models a player in the the CEO seat. Here we disable
    passing to make the tests easier.
    """

    # Create CardValue objects for ease of use later
    cv0 = CardValue(0)
    cv1 = CardValue(1)
    cv2 = CardValue(2)
    cv3 = CardValue(3)
    cv4 = CardValue(4)
    cv5 = CardValue(5)
    cv6 = CardValue(6)

    # Make the hands. Note that we disable passing below
    hand1 = Hand()
    hand1.add_cards(cv0, 1)
    hand1.add_cards(cv3, 2)
    hand1.add_cards(cv6, 4)

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

    # action: Lead highest cv6
    behavior2.value_to_play.append(None)
    behavior3.value_to_play.append(None)
    behavior4.value_to_play.append(None)

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

    # Three possible cards to play
    assert env.action_space == ActionSpaceFactory.action_space_lead

    # Lead highest
    action = env.action_space.find_full_action(ActionEnum.PLAY_HIGHEST_NUM)
    observation, reward, done, _truncated, info = env.step(action)

    assert env.action_space == ActionSpaceFactory.action_space_two_legal_lead
    assert not done
    assert reward == 0

    # Lead lowest
    action = env.action_space.find_full_action(ActionEnum.PLAY_LOWEST_WITHOUT_BREAK_NUM)
    observation, reward, done, _truncated, info = env.step(action)

    assert env.action_space == ActionSpaceFactory.action_space_one_legal_play
    assert not done
    assert reward == 0

    action = env.action_space.find_full_action(ActionEnum.PLAY_HIGHEST_NUM)
    observation, reward, done, _truncated, info = env.step(action)

    assert reward > 0
    assert done


def test_seatceoenv_observation():
    """
    Test the observation returned by SetCEOEnv.
    """

    # Create CardValue objects for ease of use later
    cv0 = CardValue(0)
    cv1 = CardValue(1)
    cv2 = CardValue(2)
    cv3 = CardValue(3)
    cv5 = CardValue(5)
    cv6 = CardValue(6)
    cv7 = CardValue(7)
    cv8 = CardValue(8)

    # Make the hands
    hand1 = Hand()
    hand1.add_cards(cv0, 1)
    hand1.add_cards(cv1, 1)
    hand1.add_cards(cv2, 3)
    hand1.add_cards(cv8, 4)

    hand2 = Hand()
    hand2.add_cards(cv1, 1)
    hand2.add_cards(cv2, 1)
    hand2.add_cards(cv7, 1)
    hand2.add_cards(cv6, 3)

    hand3 = Hand()
    hand3.add_cards(cv2, 1)
    hand3.add_cards(cv3, 1)
    hand3.add_cards(cv5, 2)
    hand3.add_cards(cv7, 3)

    hand4 = Hand()
    hand4.add_cards(cv3, 1)
    hand4.add_cards(cv0, 1)
    hand4.add_cards(cv6, 2)
    hand4.add_cards(cv8, 5)

    hands = [hand1, hand2, hand3, hand4]

    # Make the players
    behavior2 = MockPlayerBehavior()
    behavior3 = MockPlayerBehavior()
    behavior4 = MockPlayerBehavior()

    # action: Lead lowest = cv0
    behavior2.value_to_play.append(cv1)
    behavior3.value_to_play.append(cv2)
    behavior4.value_to_play.append(cv3)

    behavior4.value_to_play.append(cv0)
    # action: Play lowest = cv1
    behavior2.value_to_play.append(cv2)
    behavior3.value_to_play.append(cv3)

    behavior3.value_to_play.append(cv5)
    behavior4.value_to_play.append(cv6)

    behaviors = [behavior2, behavior3, behavior4]

    env = SeatCEOEnv(
        num_players=4,
        behaviors=behaviors,
        hands=hands,
        listener=PrintAllEventListener(),
        skip_passing=True,
    )

    observation_factory = env.observation_factory

    observation_array, _ = env.reset()
    observation = observation_factory.create_observation(array=observation_array)

    assert observation.get_starting_player() == 0
    assert observation.get_cur_trick_count() == 0
    assert observation.get_cur_trick_value() is None
    assert observation.get_last_player() is None

    assert observation.get_card_count(0) == 1
    assert observation.get_card_count(1) == 1
    assert observation.get_card_count(2) == 3
    assert observation.get_card_count(8) == 4

    assert observation.get_other_player_card_count(0) == 6
    assert observation.get_other_player_card_count(1) == 7
    assert observation.get_other_player_card_count(2) == 9

    # Lead lowest
    action = env.action_space.find_full_action(ActionEnum.PLAY_LOWEST_WITHOUT_BREAK_NUM)
    observation_array, reward, done, _truncated, info = env.step(action)

    observation = observation_factory.create_observation(array=observation_array)

    assert observation.get_starting_player() == 3
    assert observation.get_cur_trick_count() == 1
    assert observation.get_cur_trick_value() == 0
    assert observation.get_last_player() == 3

    assert observation.get_card_count(0) == 0
    assert observation.get_card_count(1) == 1
    assert observation.get_card_count(2) == 3
    assert observation.get_card_count(8) == 4

    assert observation.get_other_player_card_count(0) == 5
    assert observation.get_other_player_card_count(1) == 6
    assert observation.get_other_player_card_count(2) == 7

    # Play lowest
    action = env.action_space.find_full_action(ActionEnum.PLAY_LOWEST_WITHOUT_BREAK_NUM)
    observation_array, reward, done, _truncated, info = env.step(action)

    observation = observation_factory.create_observation(array=observation_array)

    assert observation.get_starting_player() == 2
    assert observation.get_cur_trick_count() == 2
    assert observation.get_cur_trick_value() == 6
    assert observation.get_last_player() == 3

    assert observation.get_card_count(0) == 0
    assert observation.get_card_count(1) == 0
    assert observation.get_card_count(2) == 3
    assert observation.get_card_count(8) == 4

    assert observation.get_other_player_card_count(0) == 4
    assert observation.get_other_player_card_count(1) == 3
    assert observation.get_other_player_card_count(2) == 5


def test_seatceoenv_observation_valid_plays():
    """
    Test the observation returned by SetCEOEnv. Here the observation is configured to
    include which actions are valid.
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

    # Make the hands
    hand1 = Hand()
    hand1.add_cards(cv0, 1)
    hand1.add_cards(cv1, 2)
    hand1.add_cards(cv2, 3)
    hand1.add_cards(cv4, 2)
    hand1.add_cards(cv8, 4)
    hand1.add_cards(cv9, 1)

    hand2 = Hand()
    hand2.add_cards(cv1, 1)
    hand2.add_cards(cv2, 2)
    hand2.add_cards(cv7, 1)
    hand2.add_cards(cv6, 3)

    hand3 = Hand()
    hand3.add_cards(cv2, 1)
    hand3.add_cards(cv3, 2)
    hand3.add_cards(cv5, 2)
    hand3.add_cards(cv7, 3)

    hand4 = Hand()
    hand4.add_cards(cv3, 1)
    hand4.add_cards(cv0, 2)
    hand4.add_cards(cv6, 2)
    hand4.add_cards(cv8, 5)

    hands = [hand1, hand2, hand3, hand4]

    # Make the players
    behavior2 = MockPlayerBehavior()
    behavior3 = MockPlayerBehavior()
    behavior4 = MockPlayerBehavior()

    # action: Lead lowest = cv0
    behavior2.value_to_play.append(cv1)
    behavior3.value_to_play.append(cv2)
    behavior4.value_to_play.append(cv3)

    behavior4.value_to_play.append(cv0)
    # action: Play lowest = cv1
    behavior2.value_to_play.append(cv2)
    behavior3.value_to_play.append(cv3)

    behavior3.value_to_play.append(cv5)
    behavior4.value_to_play.append(cv6)

    behaviors = [behavior2, behavior3, behavior4]

    obs_kwargs = {"include_valid_actions": True}

    env = SeatCEOEnv(
        num_players=4,
        behaviors=behaviors,
        hands=hands,
        listener=PrintAllEventListener(),
        skip_passing=True,
        obs_kwargs=obs_kwargs,
    )

    observation_factory = env.observation_factory

    observation_array, _ = env.reset()
    observation = observation_factory.create_observation(array=observation_array)

    assert observation.get_starting_player() == 0
    assert observation.get_cur_trick_count() == 0
    assert observation.get_cur_trick_value() is None
    assert observation.get_last_player() is None

    assert observation.get_card_count(0) == 1
    assert observation.get_card_count(1) == 2
    assert observation.get_card_count(2) == 3
    assert observation.get_card_count(4) == 2
    assert observation.get_card_count(8) == 4
    assert observation.get_card_count(9) == 1

    assert observation.get_other_player_card_count(0) == 7
    assert observation.get_other_player_card_count(1) == 8
    assert observation.get_other_player_card_count(2) == 10

    assert not observation.get_pass_action_valid()
    assert observation.get_play_card_action_valid(cv0.value)
    assert observation.get_play_card_action_valid(cv1.value)
    assert observation.get_play_card_action_valid(cv2.value)
    assert not observation.get_play_card_action_valid(cv3.value)
    assert observation.get_play_card_action_valid(cv4.value)
    assert not observation.get_play_card_action_valid(cv5.value)
    assert not observation.get_play_card_action_valid(cv6.value)
    assert not observation.get_play_card_action_valid(cv7.value)
    assert observation.get_play_card_action_valid(cv8.value)
    assert observation.get_play_card_action_valid(cv9.value)
    assert not observation.get_play_card_action_valid(cv10.value)

    # Lead lowest
    action = env.action_space.find_full_action(ActionEnum.PLAY_LOWEST_WITHOUT_BREAK_NUM)
    observation_array, reward, done, _truncated, info = env.step(action)

    observation = observation_factory.create_observation(array=observation_array)

    assert observation.get_starting_player() == 3
    assert observation.get_cur_trick_count() == 2
    assert observation.get_cur_trick_value() == 0
    assert observation.get_last_player() == 3

    assert observation.get_card_count(0) == 0
    assert observation.get_card_count(1) == 2
    assert observation.get_card_count(2) == 3
    assert observation.get_card_count(4) == 2
    assert observation.get_card_count(8) == 4
    assert observation.get_card_count(9) == 1

    assert observation.get_other_player_card_count(0) == 6
    assert observation.get_other_player_card_count(1) == 7
    assert observation.get_other_player_card_count(2) == 7

    assert observation.get_pass_action_valid()
    assert not observation.get_play_card_action_valid(cv0.value)
    assert observation.get_play_card_action_valid(cv1.value)
    assert observation.get_play_card_action_valid(cv2.value)
    assert not observation.get_play_card_action_valid(cv3.value)
    assert observation.get_play_card_action_valid(cv4.value)
    assert not observation.get_play_card_action_valid(cv5.value)
    assert not observation.get_play_card_action_valid(cv6.value)
    assert not observation.get_play_card_action_valid(cv7.value)
    assert observation.get_play_card_action_valid(cv8.value)
    assert not observation.get_play_card_action_valid(cv9.value)
    assert not observation.get_play_card_action_valid(cv10.value)

    # Play lowest
    action = env.action_space.find_full_action(ActionEnum.PLAY_LOWEST_WITHOUT_BREAK_NUM)
    observation_array, reward, done, _truncated, info = env.step(action)

    observation = observation_factory.create_observation(array=observation_array)

    assert observation.get_starting_player() == 2
    assert observation.get_cur_trick_count() == 2
    assert observation.get_cur_trick_value() == 6
    assert observation.get_last_player() == 3

    assert observation.get_card_count(0) == 0
    assert observation.get_card_count(1) == 0
    assert observation.get_card_count(2) == 3
    assert observation.get_card_count(4) == 2
    assert observation.get_card_count(8) == 4
    assert observation.get_card_count(9) == 1

    assert observation.get_other_player_card_count(0) == 4
    assert observation.get_other_player_card_count(1) == 3
    assert observation.get_other_player_card_count(2) == 5

    assert observation.get_pass_action_valid()
    assert not observation.get_play_card_action_valid(cv0.value)
    assert not observation.get_play_card_action_valid(cv1.value)
    assert not observation.get_play_card_action_valid(cv2.value)
    assert not observation.get_play_card_action_valid(cv3.value)
    assert not observation.get_play_card_action_valid(cv4.value)
    assert not observation.get_play_card_action_valid(cv5.value)
    assert not observation.get_play_card_action_valid(cv6.value)
    assert not observation.get_play_card_action_valid(cv7.value)
    assert observation.get_play_card_action_valid(cv8.value)
    assert not observation.get_play_card_action_valid(cv9.value)
    assert not observation.get_play_card_action_valid(cv10.value)


def test_seatceoenv_reward_cards_left():
    """
    Test the reward from losing when the environment is configured to
    include the cards left in the hand.
    """

    # Create CardValue objects for ease of use later
    cv0 = CardValue(0)
    cv1 = CardValue(1)
    cv2 = CardValue(2)
    cv3 = CardValue(3)

    # Make the hands
    hand1 = Hand()
    hand1.add_cards(cv0, 1)
    hand1.add_cards(cv1, 1)
    hand1.add_cards(cv2, 1)

    hand2 = Hand()
    hand2.add_cards(cv1, 1)

    hand3 = Hand()
    hand3.add_cards(cv2, 1)

    hand4 = Hand()
    hand4.add_cards(cv3, 1)

    hands = [hand1, hand2, hand3, hand4]

    # Make the players
    behavior2 = MockPlayerBehavior()
    behavior3 = MockPlayerBehavior()
    behavior4 = MockPlayerBehavior()

    # action: Lead lowest = cv0
    behavior2.value_to_play.append(cv1)
    behavior3.value_to_play.append(cv2)
    behavior4.value_to_play.append(cv3)

    behaviors = [behavior2, behavior3, behavior4]

    env = SeatCEOEnv(
        num_players=4,
        behaviors=behaviors,
        hands=hands,
        listener=PrintAllEventListener(),
        skip_passing=True,
        reward_includes_cards_left=True,
    )

    observation_array = env.reset()

    # Lead lowest
    action = env.action_space.find_full_action(ActionEnum.PLAY_LOWEST_WITHOUT_BREAK_NUM)
    observation_array, reward, done, _truncated, info = env.step(action)

    assert done

    cards_left = 2.0
    assert reward == pytest.approx(-1.0 - cards_left / 13.0)


def test_seatceoenv_actionspace_play_singlecard():
    """
    Test that the action space changes based on the cards available to play.
    Here we test where there is a single card value that can be played on
    a trick.
    """

    # Create CardValue objects for ease of use later
    cv0 = CardValue(0)
    cv1 = CardValue(1)
    cv2 = CardValue(2)
    cv3 = CardValue(3)
    cv4 = CardValue(4)
    cv5 = CardValue(5)
    cv6 = CardValue(6)

    # Test where there is a single card that can be played
    hand1 = Hand()
    hand1.add_cards(cv0, 1)
    hand1.add_cards(cv3, 2)

    hand2 = Hand()
    hand2.add_cards(cv1, 1)
    hand2.add_cards(cv4, 2)
    hand2.add_cards(cv6, 3)

    hand3 = Hand()
    hand3.add_cards(cv2, 1)
    hand3.add_cards(cv5, 2)
    hand3.add_cards(cv4, 3)

    hand4 = Hand()
    hand4.add_cards(cv3, 1)
    hand4.add_cards(cv2, 2)
    hand4.add_cards(cv5, 3)

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
    # action: Play single card = cv3
    behavior2.value_to_play.append(cv4)
    behavior3.value_to_play.append(cv5)

    behavior3.value_to_play.append(cv4)
    behavior4.value_to_play.append(cv5)
    behavior2.value_to_play.append(cv6)

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
    assert env.action_space == ActionSpaceFactory.action_space_two_legal_lead
    action = 1
    assert env.action_space.actions[action] == ActionEnum.PLAY_LOWEST_WITHOUT_BREAK_NUM
    observation, reward, done, _truncated, info = env.step(action)

    assert not done
    assert reward == 0

    assert env.action_space == ActionSpaceFactory.action_space_one_legal_play
    assert env.action_space.n == 2
    action = 0
    assert env.action_space.actions[action] != ActionEnum.PASS_ON_TRICK_NUM
    observation, reward, done, _truncated, info = env.step(action)

    assert reward > 0
    assert done


def test_seatceoenv_actionspace_play_twocards():
    """
    Test that the action space changes based on the cards available to play.
    Here we test where there are two card values that can be played on
    a trick.
    """

    # Create CardValue objects for ease of use later
    cv0 = CardValue(0)
    cv1 = CardValue(1)
    cv2 = CardValue(2)
    cv3 = CardValue(3)
    cv4 = CardValue(4)
    cv5 = CardValue(5)
    cv6 = CardValue(6)
    cv9 = CardValue(9)

    # Test where there are two cards that can be played
    hand1 = Hand()
    hand1.add_cards(cv0, 1)
    hand1.add_cards(cv3, 2)
    hand1.add_cards(cv9, 2)

    hand2 = Hand()
    hand2.add_cards(cv1, 1)
    hand2.add_cards(cv4, 2)
    hand2.add_cards(cv6, 3)

    hand3 = Hand()
    hand3.add_cards(cv2, 1)
    hand3.add_cards(cv5, 2)
    hand3.add_cards(cv4, 3)

    hand4 = Hand()
    hand4.add_cards(cv3, 1)
    hand4.add_cards(cv2, 2)
    hand4.add_cards(cv5, 3)

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
    # action: Play double card = cv9
    behavior2.value_to_play.append(None)
    behavior3.value_to_play.append(None)

    # action: Lead cv3
    behavior2.value_to_play.append(cv4)
    behavior3.value_to_play.append(cv5)
    behavior4.value_to_play.append(None)

    behavior3.value_to_play.append(cv4)
    behavior4.value_to_play.append(cv5)
    behavior2.value_to_play.append(cv6)

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
    assert env.action_space == ActionSpaceFactory.action_space_lead
    action = 2
    assert env.action_space.actions[action] == ActionEnum.PLAY_LOWEST_WITHOUT_BREAK_NUM
    observation, reward, done, _truncated, info = env.step(action)

    assert not done
    assert reward == 0

    # Play highest
    assert env.action_space == ActionSpaceFactory.action_space_two_legal_play
    assert env.action_space.n == 3
    action = 0
    assert env.action_space.actions[action] == ActionEnum.PLAY_HIGHEST_NUM
    observation, reward, done, _truncated, info = env.step(action)

    assert not done
    assert reward == 0

    # Lead only card
    assert env.action_space == ActionSpaceFactory.action_space_one_legal_lead
    action = 0
    observation, reward, done, _truncated, info = env.step(action)

    assert reward > 0
    assert done


def test_seatceoenv_actionspace_lead_twocards():
    """
    Test that the action space changes based on the cards available to play.
    Here we test where there are two card values that can be lead.
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

    # Test where there are two cards that can be played
    hand1 = Hand()
    hand1.add_cards(cv0, 1)
    hand1.add_cards(cv8, 2)
    hand1.add_cards(cv9, 2)
    hand1.add_cards(cv10, 2)

    hand2 = Hand()
    hand2.add_cards(cv1, 1)
    hand2.add_cards(cv4, 2)
    hand2.add_cards(cv6, 3)

    hand3 = Hand()
    hand3.add_cards(cv2, 1)
    hand3.add_cards(cv5, 2)
    hand3.add_cards(cv7, 3)

    hand4 = Hand()
    hand4.add_cards(cv3, 1)
    hand4.add_cards(cv2, 2)
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
    # action: Play cv8
    behavior2.value_to_play.append(None)
    behavior3.value_to_play.append(None)

    # action: Lead cv10
    behavior2.value_to_play.append(None)
    behavior3.value_to_play.append(None)
    behavior4.value_to_play.append(None)

    # action: Lead cv9
    behavior2.value_to_play.append(None)
    behavior3.value_to_play.append(None)
    behavior4.value_to_play.append(None)

    behavior2.value_to_play.append(cv6)
    behavior3.value_to_play.append(cv7)
    behavior4.value_to_play.append(cv8)

    behavior2.value_to_play.append(cv4)
    behavior3.value_to_play.append(cv5)

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
    assert env.action_space == ActionSpaceFactory.action_space_lead
    action = 2
    assert env.action_space.actions[action] == ActionEnum.PLAY_LOWEST_WITHOUT_BREAK_NUM
    observation, reward, done, _truncated, info = env.step(action)

    assert not done
    assert reward == 0

    # Play lowest
    assert env.action_space == ActionSpaceFactory.action_space_play
    assert env.action_space.n == 4
    action = 2
    assert env.action_space.actions[action] == ActionEnum.PLAY_LOWEST_WITHOUT_BREAK_NUM
    observation, reward, done, _truncated, info = env.step(action)

    assert not done
    assert reward == 0

    # Lead highest card
    assert env.action_space == ActionSpaceFactory.action_space_two_legal_lead
    action = 0
    assert env.action_space.actions[action] == ActionEnum.PLAY_HIGHEST_NUM
    observation, reward, done, _truncated, info = env.step(action)

    assert not done
    assert reward == 0

    # Lead only card
    assert env.action_space == ActionSpaceFactory.action_space_one_legal_lead
    action = 0
    observation, reward, done, _truncated, info = env.step(action)

    assert reward > 0
    assert done


def test_seatceoenv_cannotplay_twotricks():
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
    action = env.action_space.find_full_action(ActionEnum.PLAY_LOWEST_WITHOUT_BREAK_NUM)
    observation, reward, done, _truncated, info = env.step(action)

    assert reward < 0
    assert done


def test_seatceoenv_cannotplay_threetricks():
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
    action = env.action_space.find_full_action(ActionEnum.PLAY_LOWEST_WITHOUT_BREAK_NUM)
    observation, reward, done, _truncated, info = env.step(action)

    assert done
    assert reward < 0


def test_seatceoenv_get_afterstate():
    """
    Test SetCEOEnv.get_afterstate(). This is an end-to-end test where the environment
    creates the observation.
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
    hand1.add_cards(cv2, 3)
    hand1.add_cards(cv3, 4)

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
    hand4.add_cards(cv0, 1)
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

    behavior4.value_to_play.append(cv0)

    behaviors = [behavior2, behavior3, behavior4]

    env = SeatCEOEnv(
        num_players=4,
        behaviors=behaviors,
        hands=hands,
        listener=PrintAllEventListener(),
        skip_passing=True,
    )

    observation_factory = env.observation_factory

    observation_array, _ = env.reset()
    observation = observation_factory.create_observation(array=observation_array)

    assert observation.get_card_count(0) == 1
    assert observation.get_card_count(1) == 2
    assert observation.get_card_count(2) == 3
    assert observation.get_card_count(3) == 4

    assert observation.get_last_player() is None

    # Test afterstate after lead highest
    action = env.action_space.find_full_action(ActionEnum.PLAY_HIGHEST_NUM)
    afterstate_array, played_card = env.get_afterstate(observation_array, action)
    assert played_card == cv3
    afterstate = observation_factory.create_observation(array=afterstate_array)

    assert afterstate.get_card_count(0) == 1
    assert afterstate.get_card_count(1) == 2
    assert afterstate.get_card_count(2) == 3
    assert afterstate.get_card_count(3) == 0

    assert afterstate.get_last_player() == 0

    # Test afterstate after lead lowest
    action = env.action_space.find_full_action(ActionEnum.PLAY_LOWEST_WITHOUT_BREAK_NUM)
    afterstate_array, played_card = env.get_afterstate(observation_array, action)
    assert played_card == cv0
    afterstate = observation_factory.create_observation(array=afterstate_array)

    assert afterstate.get_card_count(0) == 0
    assert afterstate.get_card_count(1) == 2
    assert afterstate.get_card_count(2) == 3
    assert afterstate.get_card_count(3) == 4

    assert afterstate.get_last_player() == 0

    # Lead lowest
    action = env.action_space.find_full_action(ActionEnum.PLAY_LOWEST_WITHOUT_BREAK_NUM)
    observation_array, reward, done, _truncated, info = env.step(action)

    observation = observation_factory.create_observation(array=observation_array)

    assert observation.get_card_count(0) == 0
    assert observation.get_card_count(1) == 2
    assert observation.get_card_count(2) == 3
    assert observation.get_card_count(3) == 4

    assert observation.get_last_player() == 3

    # Test afterstate after play highest
    action = env.action_space.find_full_action(ActionEnum.PLAY_HIGHEST_NUM)
    afterstate_array, played_card = env.get_afterstate(observation_array, action)
    afterstate = observation_factory.create_observation(array=afterstate_array)

    assert afterstate.get_card_count(0) == 0
    assert afterstate.get_card_count(1) == 2
    assert afterstate.get_card_count(2) == 3
    assert afterstate.get_card_count(3) == 3

    assert afterstate.get_last_player() == 0

    # Test afterstate after pass
    action = env.action_space.find_full_action(ActionEnum.PASS_ON_TRICK_NUM)
    afterstate_array, played_card = env.get_afterstate(observation_array, action)
    assert played_card is None
    afterstate = observation_factory.create_observation(array=afterstate_array)

    assert afterstate.get_card_count(0) == 0
    assert afterstate.get_card_count(1) == 2
    assert afterstate.get_card_count(2) == 3
    assert afterstate.get_card_count(3) == 4

    assert afterstate.get_last_player() == 3


def test_seatceoenv_cardactionspace():
    """
    Test the environment that models a player in the the CEO seat. Test where the
    enviroment uses an action space corresponding to the cards that can be played.
    """

    # Create CardValue objects for ease of use later
    cv0 = CardValue(0)
    cv1 = CardValue(1)
    cv2 = CardValue(2)
    cv3 = CardValue(3)
    cv4 = CardValue(4)
    cv5 = CardValue(5)
    cv6 = CardValue(6)

    # Make the hands. Note that we disable passing below
    hand1 = Hand()
    hand1.add_cards(cv0, 1)
    hand1.add_cards(cv3, 2)
    hand1.add_cards(cv6, 4)

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

    # action: Lead highest cv6
    behavior2.value_to_play.append(None)
    behavior3.value_to_play.append(None)
    behavior4.value_to_play.append(None)

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
        action_space_type="card",
        num_players=4,
        behaviors=behaviors,
        hands=hands,
        listener=PrintAllEventListener(),
        skip_passing=True,
    )

    observation_factory = env.observation_factory

    observation_array = env.reset()

    # Three possible cards to play
    assert env.action_space.n == 3

    # Lead highest
    observation_array, reward, done, _truncated, info = env.step(2)

    assert env.action_space.n == 2
    assert not done
    assert reward == 0

    observation = observation_factory.create_observation(array=observation_array)

    assert observation.get_card_count(0) == 1
    assert observation.get_card_count(1) == 0
    assert observation.get_card_count(2) == 0
    assert observation.get_card_count(3) == 2
    assert observation.get_card_count(4) == 0
    assert observation.get_card_count(5) == 0

    # Lead lowest
    observation_array, reward, done, _truncated, info = env.step(0)

    assert env.action_space.n == 1 + 1
    assert not done
    assert reward == 0

    observation = observation_factory.create_observation(array=observation_array)

    assert observation.get_card_count(0) == 0
    assert observation.get_card_count(1) == 0
    assert observation.get_card_count(2) == 0
    assert observation.get_card_count(3) == 2
    assert observation.get_card_count(4) == 0
    assert observation.get_card_count(5) == 0

    # Lead final card
    observation_array, reward, done, _truncated, info = env.step(0)

    assert reward > 0
    assert done


def test_seatceoenv_cardactionspace_notplayable():
    """
    Test the environment that models a player in the the CEO seat. Test where the
    enviroment uses an action space corresponding to the cards that can be played. Here
    there are cards in the hand that can't be played.
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

    # Make the hands. Note that we disable passing below
    hand1 = Hand()
    hand1.add_cards(cv0, 1)
    hand1.add_cards(cv3, 2)
    hand1.add_cards(cv6, 1)

    hand2 = Hand()
    hand2.add_cards(cv1, 1)
    hand2.add_cards(cv4, 2)
    hand2.add_cards(cv7, 1)
    hand2.add_cards(cv8, 1)

    hand3 = Hand()
    hand3.add_cards(cv2, 1)
    hand3.add_cards(cv5, 2)
    hand3.add_cards(cv0, 1)
    hand3.add_cards(cv9, 1)

    hand4 = Hand()
    hand4.add_cards(cv3, 1)
    hand4.add_cards(cv2, 2)
    hand4.add_cards(cv1, 1)
    hand4.add_cards(cv10, 1)

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
    # action: Play highest = cv6
    behavior2.value_to_play.append(cv7)

    behavior2.value_to_play.append(cv8)
    behavior3.value_to_play.append(cv9)
    behavior4.value_to_play.append(cv10)

    behaviors = [behavior2, behavior3, behavior4]

    env = SeatCEOEnv(
        action_space_type="card",
        num_players=4,
        behaviors=behaviors,
        hands=hands,
        listener=PrintAllEventListener(),
        skip_passing=True,
    )

    observation_factory = env.observation_factory

    observation_array = env.reset()

    # Three possible cards to play
    assert env.action_space.n == 3

    # Lead lowest
    observation_array, reward, done, _truncated, info = env.step(0)

    # Only one card can be played, since doubles were led.
    assert env.action_space.n == 1 + 1
    assert not done
    assert reward == 0

    observation = observation_factory.create_observation(array=observation_array)

    assert observation.get_cur_trick_count() == 2

    assert observation.get_card_count(0) == 0
    assert observation.get_card_count(1) == 0
    assert observation.get_card_count(2) == 0
    assert observation.get_card_count(3) == 2
    assert observation.get_card_count(4) == 0
    assert observation.get_card_count(5) == 0
    assert observation.get_card_count(6) == 1

    # Lead the only card possible
    observation_array, reward, done, _truncated, info = env.step(0)

    assert env.action_space.n == 2
    assert not done
    assert reward == 0

    observation = observation_factory.create_observation(array=observation_array)

    assert observation.get_card_count(0) == 0
    assert observation.get_card_count(1) == 0
    assert observation.get_card_count(2) == 0
    assert observation.get_card_count(3) == 0
    assert observation.get_card_count(4) == 0
    assert observation.get_card_count(5) == 0
    assert observation.get_card_count(6) == 1

    # Lead final card
    observation_array, reward, done, _truncated, info = env.step(0)

    assert reward > 0
    assert done


def test_seatceoenv_cardactionspace_pass_on_trick():
    """
    Test the environment that models a player in the the CEO seat. Test where the
    enviroment uses an action space corresponding to the cards that can be played. Here
    we test the pass action."""

    # Create CardValue objects for ease of use later
    cv0 = CardValue(0)
    cv1 = CardValue(1)
    cv2 = CardValue(2)
    cv3 = CardValue(3)
    cv4 = CardValue(4)
    cv6 = CardValue(6)
    cv7 = CardValue(7)
    cv8 = CardValue(8)
    cv9 = CardValue(9)
    cv10 = CardValue(10)

    # Make the hands. Note that we disable passing before the trick below.
    hand1 = Hand()
    hand1.add_cards(cv0, 1)
    hand1.add_cards(cv6, 1)

    hand2 = Hand()
    hand2.add_cards(cv1, 1)
    hand2.add_cards(cv3, 1)
    hand2.add_cards(cv7, 1)
    hand2.add_cards(cv8, 1)

    hand3 = Hand()
    hand3.add_cards(cv2, 1)
    hand3.add_cards(cv4, 1)
    hand3.add_cards(cv0, 1)
    hand3.add_cards(cv9, 1)

    hand4 = Hand()
    hand4.add_cards(cv3, 1)
    hand4.add_cards(cv2, 1)
    hand4.add_cards(cv1, 1)
    hand4.add_cards(cv10, 1)

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
    # action: Pass
    behavior2.value_to_play.append(cv3)
    behavior3.value_to_play.append(cv4)

    behavior3.value_to_play.append(cv0)
    behavior4.value_to_play.append(cv1)
    # action: Play highest = cv6
    behavior2.value_to_play.append(cv7)

    behavior2.value_to_play.append(cv8)
    behavior3.value_to_play.append(cv9)
    behavior4.value_to_play.append(cv10)

    behaviors = [behavior2, behavior3, behavior4]

    env = SeatCEOEnv(
        action_space_type="card",
        num_players=4,
        behaviors=behaviors,
        hands=hands,
        listener=PrintAllEventListener(),
        skip_passing=True,
    )

    observation_factory = env.observation_factory

    observation_array, _ = env.reset()
    observation = observation_factory.create_observation(array=observation_array)

    # Three possible cards to play
    assert env.action_space.n == 2

    # Lead lowest
    observation_array, reward, done, _truncated, info = env.step(0)

    # Only one card can be played.
    assert env.action_space.n == 1 + 1
    assert not done
    assert reward == 0

    observation = observation_factory.create_observation(array=observation_array)

    assert observation.get_cur_trick_count() == 1

    assert observation.get_card_count(0) == 0
    assert observation.get_card_count(1) == 0
    assert observation.get_card_count(2) == 0
    assert observation.get_card_count(3) == 0
    assert observation.get_card_count(4) == 0
    assert observation.get_card_count(5) == 0
    assert observation.get_card_count(6) == 1

    # Pass
    observation_array, reward, done, _truncated, info = env.step(1)

    assert env.action_space.n == 1 + 1
    assert not done
    assert reward == 0

    observation = observation_factory.create_observation(array=observation_array)

    assert observation.get_card_count(0) == 0
    assert observation.get_card_count(1) == 0
    assert observation.get_card_count(2) == 0
    assert observation.get_card_count(3) == 0
    assert observation.get_card_count(4) == 0
    assert observation.get_card_count(5) == 0
    assert observation.get_card_count(6) == 1

    # Lead final card
    observation_array, reward, done, _truncated, info = env.step(0)

    assert reward > 0
    assert done


def test_seatceoenv_cardactionspace_get_afterstate():
    """
    Test SetCEOEnv.get_afterstate() when the environment is using card action spaces.
    This is an end-to-end test where the environment creates the observation.
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
    hand1.add_cards(cv2, 3)
    hand1.add_cards(cv3, 4)

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
    hand4.add_cards(cv0, 1)
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

    behavior4.value_to_play.append(cv0)

    behaviors = [behavior2, behavior3, behavior4]

    env = SeatCEOEnv(
        action_space_type="card",
        num_players=4,
        behaviors=behaviors,
        hands=hands,
        listener=PrintAllEventListener(),
        skip_passing=True,
    )

    observation_factory = env.observation_factory

    observation_array, _ = env.reset()
    observation = observation_factory.create_observation(array=observation_array)

    assert observation.get_card_count(0) == 1
    assert observation.get_card_count(1) == 2
    assert observation.get_card_count(2) == 3
    assert observation.get_card_count(3) == 4

    assert observation.get_last_player() is None

    # Test afterstate after lead highest
    action = env.action_space.n - 1
    afterstate_array, played_card = env.get_afterstate(observation_array, action)
    afterstate = observation_factory.create_observation(array=afterstate_array)

    assert afterstate.get_card_count(0) == 1
    assert afterstate.get_card_count(1) == 2
    assert afterstate.get_card_count(2) == 3
    assert afterstate.get_card_count(3) == 0

    assert afterstate.get_cur_trick_count() == 4
    assert afterstate.get_cur_trick_value() == 3
    assert afterstate.get_starting_player() == 0
    assert afterstate.get_last_player() == 0

    # Test afterstate after lead lowest
    action = 0
    afterstate_array, played_card = env.get_afterstate(observation_array, action)
    afterstate = observation_factory.create_observation(array=afterstate_array)

    assert afterstate.get_card_count(0) == 0
    assert afterstate.get_card_count(1) == 2
    assert afterstate.get_card_count(2) == 3
    assert afterstate.get_card_count(3) == 4

    assert afterstate.get_cur_trick_count() == 1
    assert afterstate.get_cur_trick_value() == 0
    assert afterstate.get_starting_player() == 0
    assert afterstate.get_last_player() == 0

    # Lead lowest
    observation_array, reward, done, _truncated, info = env.step(0)

    observation = observation_factory.create_observation(array=observation_array)

    assert observation.get_card_count(0) == 0
    assert observation.get_card_count(1) == 2
    assert observation.get_card_count(2) == 3
    assert observation.get_card_count(3) == 4

    assert observation.get_cur_trick_count() == 1
    assert observation.get_cur_trick_value() == 0
    assert observation.get_starting_player() == 3
    assert observation.get_last_player() == 3

    # Test afterstate after play highest
    action = env.action_space.n - 2
    afterstate_array, played_card = env.get_afterstate(observation_array, action)
    afterstate = observation_factory.create_observation(array=afterstate_array)

    assert afterstate.get_card_count(0) == 0
    assert afterstate.get_card_count(1) == 2
    assert afterstate.get_card_count(2) == 3
    assert afterstate.get_card_count(3) == 3

    assert afterstate.get_cur_trick_count() == 1
    assert afterstate.get_cur_trick_value() == 3
    assert afterstate.get_starting_player() == 3
    assert afterstate.get_last_player() == 0

    # Test afterstate after pass
    action = env.action_space.n - 1
    afterstate_array, played_card = env.get_afterstate(observation_array, action)
    afterstate = observation_factory.create_observation(array=afterstate_array)

    assert afterstate.get_card_count(0) == 0
    assert afterstate.get_card_count(1) == 2
    assert afterstate.get_card_count(2) == 3
    assert afterstate.get_card_count(3) == 4

    assert afterstate.get_cur_trick_count() == 1
    assert afterstate.get_cur_trick_value() == 0
    assert afterstate.get_starting_player() == 3
    assert afterstate.get_last_player() == 3


def test_seatceoenv_cardactionspace_get_afterstate_trickstate():
    """
    Test SetCEOEnv.get_afterstate() when the environment is using card action spaces.
    Test that the trick information is correctly updated.
    """

    # Create CardValue objects for ease of use later
    cv0 = CardValue(0)
    cv1 = CardValue(1)
    cv2 = CardValue(2)
    cv3 = CardValue(3)
    CardValue(4)
    CardValue(5)
    CardValue(6)
    CardValue(7)
    CardValue(8)

    # Set up the environment.
    hand1 = Hand()
    hand1.add_cards(cv1, 1)

    hand2 = Hand()
    hand2.add_cards(cv1, 1)

    hand3 = Hand()
    hand3.add_cards(cv2, 1)

    hand4 = Hand()
    hand4.add_cards(cv3, 1)

    behavior2 = MockPlayerBehavior()
    behavior3 = MockPlayerBehavior()
    behavior4 = MockPlayerBehavior()

    hands = [hand1, hand2, hand3, hand4]
    behaviors = [behavior2, behavior3, behavior4]

    env = SeatCEOEnv(
        action_space_type="card",
        num_players=4,
        behaviors=behaviors,
        hands=hands,
        listener=PrintAllEventListener(),
        skip_passing=True,
    )

    observation_factory = env.observation_factory

    # Test when the agent leads
    hand = Hand()
    hand.add_cards(cv0, 1)
    hand.add_cards(cv1, 2)
    hand.add_cards(cv2, 3)
    hand.add_cards(cv3, 4)
    hand_card_count = 4

    hands = [hand, hand2, hand3, hand4]

    state = rd.RoundState(hands, None)

    observation = observation_factory.create_observation(
        type="lead", cur_hand=hand, starting_player=0, state=state
    )

    assert observation.get_card_count(0) == 1
    assert observation.get_card_count(1) == 2
    assert observation.get_card_count(2) == 3
    assert observation.get_card_count(3) == 4

    assert observation.get_cur_trick_count() == 0
    assert observation.get_cur_trick_value() is None
    assert observation.get_starting_player() == 0

    # Test afterstate after lead highest
    action = hand_card_count - 1
    afterstate_array, played_card = env.get_afterstate(observation.get_array(), action)
    afterstate = observation_factory.create_observation(array=afterstate_array)

    assert afterstate.get_card_count(0) == 1
    assert afterstate.get_card_count(1) == 2
    assert afterstate.get_card_count(2) == 3
    assert afterstate.get_card_count(3) == 0

    assert afterstate.get_cur_trick_count() == 4
    assert afterstate.get_cur_trick_value() == 3
    assert afterstate.get_starting_player() == 0

    # Test afterstate after lead lowest
    action = 0
    afterstate_array, played_card = env.get_afterstate(observation.get_array(), action)
    afterstate = observation_factory.create_observation(array=afterstate_array)

    assert afterstate.get_card_count(0) == 0
    assert afterstate.get_card_count(1) == 2
    assert afterstate.get_card_count(2) == 3
    assert afterstate.get_card_count(3) == 4

    assert afterstate.get_cur_trick_count() == 1
    assert afterstate.get_cur_trick_value() == 0
    assert afterstate.get_starting_player() == 0

    # Test when the agent plays on a trick.
    hand = Hand()
    hand.add_cards(cv0, 4)
    hand.add_cards(cv1, 4)
    hand.add_cards(cv2, 4)
    hand.add_cards(cv3, 4)

    hands = [hand, hand2, hand3, hand4]

    state = rd.RoundState(hands, 3)

    observation = observation_factory.create_observation(
        type="play",
        cur_hand=hand,
        starting_player=3,
        cur_card_value=cv0,
        cur_card_count=2,
        state=state,
    )

    assert observation.get_card_count(0) == 4
    assert observation.get_card_count(1) == 4
    assert observation.get_card_count(2) == 4
    assert observation.get_card_count(3) == 4

    assert observation.get_cur_trick_count() == 2
    assert observation.get_cur_trick_value() == 0
    assert observation.get_starting_player() == 3
    assert observation.get_last_player() == 3

    # Test afterstate after play highest
    action = hand_card_count - 2
    afterstate_array, played_card = env.get_afterstate(observation.get_array(), action)
    afterstate = observation_factory.create_observation(array=afterstate_array)

    assert afterstate.get_card_count(0) == 4
    assert afterstate.get_card_count(1) == 4
    assert afterstate.get_card_count(2) == 4
    assert afterstate.get_card_count(3) == 2

    assert afterstate.get_cur_trick_count() == 2
    assert afterstate.get_cur_trick_value() == 3
    assert afterstate.get_starting_player() == 3
    assert afterstate.get_last_player() == 0

    # Test afterstate after play lowest
    action = 0
    afterstate_array, played_card = env.get_afterstate(observation.get_array(), action)
    afterstate = observation_factory.create_observation(array=afterstate_array)

    assert afterstate.get_card_count(0) == 4
    assert afterstate.get_card_count(1) == 2
    assert afterstate.get_card_count(2) == 4
    assert afterstate.get_card_count(3) == 4

    assert afterstate.get_cur_trick_count() == 2
    assert afterstate.get_cur_trick_value() == 1
    assert afterstate.get_starting_player() == 3
    assert afterstate.get_last_player() == 0

    # Test afterstate after pass
    action = hand_card_count - 1
    afterstate_array, played_card = env.get_afterstate(observation.get_array(), action)
    afterstate = observation_factory.create_observation(array=afterstate_array)

    assert afterstate.get_card_count(0) == 4
    assert afterstate.get_card_count(1) == 4
    assert afterstate.get_card_count(2) == 4
    assert afterstate.get_card_count(3) == 4

    assert afterstate.get_cur_trick_count() == 2
    assert afterstate.get_cur_trick_value() == 0
    assert afterstate.get_starting_player() == 3
    assert afterstate.get_last_player() == 3
