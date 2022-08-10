import pytest
import CEO.cards.deck as deck
from CEO.cards.hand import *
import CEO.cards.round as rd
import CEO.cards.player as player
from gym_ceo.envs.actions import (
    Actions,
    ActionSpaceFactory,
    AllCardActionSpaceFactory,
    CEOActionSpace,
)


def test_play_lowest():
    """
    Test Actions.play_lowest
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

    # Create the object
    actions = Actions()

    # Test when a single is led
    hand = Hand()
    hand.add_cards(cv0, 1)
    hand.add_cards(cv1, 1)
    hand.add_cards(cv6, 1)
    hand.add_cards(cv7, 1)

    assert actions.play_lowest(hand, cv0, 1) == cv1
    assert actions.play_lowest(hand, cv1, 1) == cv6
    assert actions.play_lowest(hand, cv2, 1) == cv6
    assert actions.play_lowest(hand, cv3, 1) == cv6
    assert actions.play_lowest(hand, cv4, 1) == cv6
    assert actions.play_lowest(hand, cv5, 1) == cv6
    assert actions.play_lowest(hand, cv6, 1) == cv7
    assert actions.play_lowest(hand, cv7, 1) == None
    assert actions.play_lowest(hand, cv8, 1) == None

    # Test when a pair is led
    hand = Hand()
    hand.add_cards(cv0, 2)
    hand.add_cards(cv1, 2)
    hand.add_cards(cv6, 1)
    hand.add_cards(cv7, 2)

    assert actions.play_lowest(hand, cv0, 2) == cv1
    assert actions.play_lowest(hand, cv1, 2) == cv7
    assert actions.play_lowest(hand, cv2, 2) == cv7
    assert actions.play_lowest(hand, cv3, 2) == cv7
    assert actions.play_lowest(hand, cv4, 2) == cv7
    assert actions.play_lowest(hand, cv5, 2) == cv7
    assert actions.play_lowest(hand, cv6, 2) == cv7
    assert actions.play_lowest(hand, cv7, 2) == None
    assert actions.play_lowest(hand, cv8, 2) == None


def test_play_lowest_without_breaking_sets():
    """
    Test Actions.play_lowest_without_breaking_sets
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

    # Create the object
    actions = Actions()

    # Test when a single is led
    hand = Hand()
    hand.add_cards(cv0, 1)
    hand.add_cards(cv1, 1)
    hand.add_cards(cv6, 1)
    hand.add_cards(cv7, 1)

    assert actions.play_lowest_without_breaking_sets(hand, cv0, 1) == cv1
    assert actions.play_lowest_without_breaking_sets(hand, cv1, 1) == cv6
    assert actions.play_lowest_without_breaking_sets(hand, cv2, 1) == cv6
    assert actions.play_lowest_without_breaking_sets(hand, cv3, 1) == cv6
    assert actions.play_lowest_without_breaking_sets(hand, cv4, 1) == cv6
    assert actions.play_lowest_without_breaking_sets(hand, cv5, 1) == cv6
    assert actions.play_lowest_without_breaking_sets(hand, cv6, 1) == cv7
    assert actions.play_lowest_without_breaking_sets(hand, cv7, 1) == None
    assert actions.play_lowest_without_breaking_sets(hand, cv8, 1) == None

    # Test when a pair is led
    hand = Hand()
    hand.add_cards(cv0, 2)
    hand.add_cards(cv1, 2)
    hand.add_cards(cv6, 1)
    hand.add_cards(cv7, 2)

    assert actions.play_lowest_without_breaking_sets(hand, cv0, 2) == cv1
    assert actions.play_lowest_without_breaking_sets(hand, cv1, 2) == cv7
    assert actions.play_lowest_without_breaking_sets(hand, cv2, 2) == cv7
    assert actions.play_lowest_without_breaking_sets(hand, cv3, 2) == cv7
    assert actions.play_lowest_without_breaking_sets(hand, cv4, 2) == cv7
    assert actions.play_lowest_without_breaking_sets(hand, cv5, 2) == cv7
    assert actions.play_lowest_without_breaking_sets(hand, cv6, 2) == cv7
    assert actions.play_lowest_without_breaking_sets(hand, cv7, 2) == None
    assert actions.play_lowest_without_breaking_sets(hand, cv8, 2) == None

    # Test when a pair is led and there is a triple
    hand = Hand()
    hand.add_cards(cv0, 2)
    hand.add_cards(cv1, 3)
    hand.add_cards(cv6, 1)
    hand.add_cards(cv7, 2)

    assert actions.play_lowest_without_breaking_sets(hand, cv0, 2) == cv7
    assert actions.play_lowest_without_breaking_sets(hand, cv1, 2) == cv7
    assert actions.play_lowest_without_breaking_sets(hand, cv2, 2) == cv7
    assert actions.play_lowest_without_breaking_sets(hand, cv3, 2) == cv7
    assert actions.play_lowest_without_breaking_sets(hand, cv4, 2) == cv7
    assert actions.play_lowest_without_breaking_sets(hand, cv5, 2) == cv7
    assert actions.play_lowest_without_breaking_sets(hand, cv6, 2) == cv7
    assert actions.play_lowest_without_breaking_sets(hand, cv7, 2) == None
    assert actions.play_lowest_without_breaking_sets(hand, cv8, 2) == None

    hand = Hand()
    hand.add_cards(cv0, 2)
    hand.add_cards(cv4, 3)
    hand.add_cards(cv5, 2)
    hand.add_cards(cv6, 1)
    hand.add_cards(cv7, 2)

    assert actions.play_lowest_without_breaking_sets(hand, cv0, 2) == cv5
    assert actions.play_lowest_without_breaking_sets(hand, cv1, 2) == cv5
    assert actions.play_lowest_without_breaking_sets(hand, cv2, 2) == cv5
    assert actions.play_lowest_without_breaking_sets(hand, cv3, 2) == cv5
    assert actions.play_lowest_without_breaking_sets(hand, cv4, 2) == cv5
    assert actions.play_lowest_without_breaking_sets(hand, cv5, 2) == cv7
    assert actions.play_lowest_without_breaking_sets(hand, cv6, 2) == cv7
    assert actions.play_lowest_without_breaking_sets(hand, cv7, 2) == None
    assert actions.play_lowest_without_breaking_sets(hand, cv8, 2) == None


def test_play_second_lowest():
    """
    Test Actions.play_second_lowest
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

    # Create the object
    actions = Actions()

    # Test when a single is led
    hand = Hand()
    hand.add_cards(cv0, 1)
    hand.add_cards(cv1, 1)
    hand.add_cards(cv6, 1)
    hand.add_cards(cv7, 1)

    assert actions.play_second_lowest(hand, cv0, 1) == cv6
    assert actions.play_second_lowest(hand, cv1, 1) == cv7
    assert actions.play_second_lowest(hand, cv2, 1) == cv7
    assert actions.play_second_lowest(hand, cv3, 1) == cv7
    assert actions.play_second_lowest(hand, cv4, 1) == cv7
    assert actions.play_second_lowest(hand, cv5, 1) == cv7
    assert actions.play_second_lowest(hand, cv6, 1) == cv7
    assert actions.play_second_lowest(hand, cv7, 1) == None
    assert actions.play_second_lowest(hand, cv8, 1) == None

    # Test when a pair is led
    hand = Hand()
    hand.add_cards(cv0, 2)
    hand.add_cards(cv1, 2)
    hand.add_cards(cv6, 1)
    hand.add_cards(cv7, 2)
    hand.add_cards(cv8, 2)

    assert actions.play_second_lowest(hand, cv0, 2) == cv7
    assert actions.play_second_lowest(hand, cv1, 2) == cv8
    assert actions.play_second_lowest(hand, cv2, 2) == cv8
    assert actions.play_second_lowest(hand, cv3, 2) == cv8
    assert actions.play_second_lowest(hand, cv4, 2) == cv8
    assert actions.play_second_lowest(hand, cv5, 2) == cv8
    assert actions.play_second_lowest(hand, cv6, 2) == cv8
    assert actions.play_second_lowest(hand, cv7, 2) == cv8
    assert actions.play_second_lowest(hand, cv8, 2) == None


def test_play_second_lowest_without_breaking_sets():
    """
    Test Actions.play_second_lowest_without_breaking_sets
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

    # Create the object
    actions = Actions()

    # Test when a single is led
    hand = Hand()
    hand.add_cards(cv0, 1)
    hand.add_cards(cv1, 1)
    hand.add_cards(cv6, 1)
    hand.add_cards(cv7, 1)

    assert actions.play_second_lowest_without_breaking_sets(hand, cv0, 1) == cv6
    assert actions.play_second_lowest_without_breaking_sets(hand, cv1, 1) == cv7
    assert actions.play_second_lowest_without_breaking_sets(hand, cv2, 1) == cv7
    assert actions.play_second_lowest_without_breaking_sets(hand, cv3, 1) == cv7
    assert actions.play_second_lowest_without_breaking_sets(hand, cv4, 1) == cv7
    assert actions.play_second_lowest_without_breaking_sets(hand, cv5, 1) == cv7
    assert actions.play_second_lowest_without_breaking_sets(hand, cv6, 1) == cv7
    assert actions.play_second_lowest_without_breaking_sets(hand, cv7, 1) == None
    assert actions.play_second_lowest_without_breaking_sets(hand, cv8, 1) == None

    # Test when a pair is led
    hand = Hand()
    hand.add_cards(cv0, 2)
    hand.add_cards(cv1, 2)
    hand.add_cards(cv6, 1)
    hand.add_cards(cv7, 2)
    hand.add_cards(cv8, 2)

    assert actions.play_second_lowest_without_breaking_sets(hand, cv0, 2) == cv7
    assert actions.play_second_lowest_without_breaking_sets(hand, cv1, 2) == cv8
    assert actions.play_second_lowest_without_breaking_sets(hand, cv2, 2) == cv8
    assert actions.play_second_lowest_without_breaking_sets(hand, cv3, 2) == cv8
    assert actions.play_second_lowest_without_breaking_sets(hand, cv4, 2) == cv8
    assert actions.play_second_lowest_without_breaking_sets(hand, cv5, 2) == cv8
    assert actions.play_second_lowest_without_breaking_sets(hand, cv6, 2) == cv8
    assert actions.play_second_lowest_without_breaking_sets(hand, cv7, 2) == cv8
    assert actions.play_second_lowest_without_breaking_sets(hand, cv8, 2) == None

    # Test when a pair is led and there is a triple
    hand = Hand()
    hand.add_cards(cv0, 2)
    hand.add_cards(cv1, 3)
    hand.add_cards(cv6, 1)
    hand.add_cards(cv7, 2)
    hand.add_cards(cv8, 2)
    hand.add_cards(cv9, 2)

    assert actions.play_second_lowest_without_breaking_sets(hand, cv0, 2) == cv8
    assert actions.play_second_lowest_without_breaking_sets(hand, cv1, 2) == cv8
    assert actions.play_second_lowest_without_breaking_sets(hand, cv2, 2) == cv8
    assert actions.play_second_lowest_without_breaking_sets(hand, cv3, 2) == cv8
    assert actions.play_second_lowest_without_breaking_sets(hand, cv4, 2) == cv8
    assert actions.play_second_lowest_without_breaking_sets(hand, cv5, 2) == cv8
    assert actions.play_second_lowest_without_breaking_sets(hand, cv6, 2) == cv8
    assert actions.play_second_lowest_without_breaking_sets(hand, cv7, 2) == cv9
    assert actions.play_second_lowest_without_breaking_sets(hand, cv8, 2) == cv9
    assert actions.play_second_lowest_without_breaking_sets(hand, cv9, 2) == None
    assert actions.play_second_lowest_without_breaking_sets(hand, cv10, 2) == None


def test_play_highest():
    """
    Test Actions.play_highest
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

    # Create the object
    actions = Actions()

    # Test when a single is led
    hand = Hand()
    hand.add_cards(cv0, 1)
    hand.add_cards(cv1, 1)
    hand.add_cards(cv6, 1)
    hand.add_cards(cv7, 1)

    assert actions.play_highest(hand, cv0, 1) == cv7
    assert actions.play_highest(hand, cv1, 1) == cv7
    assert actions.play_highest(hand, cv2, 1) == cv7
    assert actions.play_highest(hand, cv3, 1) == cv7
    assert actions.play_highest(hand, cv4, 1) == cv7
    assert actions.play_highest(hand, cv5, 1) == cv7
    assert actions.play_highest(hand, cv6, 1) == cv7
    assert actions.play_highest(hand, cv7, 1) == None
    assert actions.play_highest(hand, cv8, 1) == None

    # Test when a pair is led
    hand = Hand()
    hand.add_cards(cv0, 2)
    hand.add_cards(cv1, 2)
    hand.add_cards(cv6, 1)
    hand.add_cards(cv7, 2)

    assert actions.play_highest(hand, cv0, 2) == cv7
    assert actions.play_highest(hand, cv1, 2) == cv7
    assert actions.play_highest(hand, cv2, 2) == cv7
    assert actions.play_highest(hand, cv3, 2) == cv7
    assert actions.play_highest(hand, cv4, 2) == cv7
    assert actions.play_highest(hand, cv5, 2) == cv7
    assert actions.play_highest(hand, cv6, 2) == cv7
    assert actions.play_highest(hand, cv7, 2) == None
    assert actions.play_highest(hand, cv8, 2) == None


def test_ActionSpaceFactory_TrickWithSingleLed():
    """Test when a trick started with a single, in particular
    where the player has low doubles.
    """

    factory = ActionSpaceFactory()

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

    # Test where there are low singles
    hand = Hand()
    hand.add_cards(cv1, 1)
    hand.add_cards(cv5, 2)
    hand.add_cards(cv6, 2)
    hand.add_cards(cv7, 1)

    action_space = factory.create_play(hand, cv0, 1)
    assert action_space == ActionSpaceFactory.action_space_two_legal_play

    hand = Hand()
    hand.add_cards(cv1, 1)
    hand.add_cards(cv5, 1)
    hand.add_cards(cv6, 2)
    hand.add_cards(cv7, 1)

    action_space = factory.create_play(hand, cv0, 1)
    assert action_space == ActionSpaceFactory.action_space_play

    # Test where there are low doubles that can't be played because we
    # currently don't break up low sets.
    hand = Hand()
    hand.add_cards(cv1, 2)
    hand.add_cards(cv5, 2)
    hand.add_cards(cv6, 2)
    hand.add_cards(cv7, 1)

    action_space = factory.create_play(hand, cv0, 1)
    assert action_space == ActionSpaceFactory.action_space_one_legal_play

    hand = Hand()
    hand.add_cards(cv1, 2)
    hand.add_cards(cv5, 2)
    hand.add_cards(cv6, 2)
    hand.add_cards(cv7, 1)
    hand.add_cards(cv8, 1)

    action_space = factory.create_play(hand, cv0, 1)
    assert action_space == ActionSpaceFactory.action_space_two_legal_play

    hand = Hand()
    hand.add_cards(cv1, 2)
    hand.add_cards(cv5, 2)
    hand.add_cards(cv6, 2)
    hand.add_cards(cv7, 1)
    hand.add_cards(cv8, 1)
    hand.add_cards(cv9, 1)

    action_space = factory.create_play(hand, cv0, 1)
    assert action_space == ActionSpaceFactory.action_space_play

    # Test that we can break up the highest set of cards
    hand = Hand()
    hand.add_cards(cv1, 2)
    hand.add_cards(cv5, 2)
    hand.add_cards(cv6, 2)
    hand.add_cards(cv7, 2)
    hand.add_cards(cv8, 2)
    hand.add_cards(cv9, 2)

    action_space = factory.create_play(hand, cv0, 1)
    assert action_space == ActionSpaceFactory.action_space_one_legal_play

    hand = Hand()
    hand.add_cards(cv1, 1)
    hand.add_cards(cv5, 2)
    hand.add_cards(cv6, 2)
    hand.add_cards(cv7, 2)
    hand.add_cards(cv8, 2)
    hand.add_cards(cv9, 2)

    action_space = factory.create_play(hand, cv0, 1)
    assert action_space == ActionSpaceFactory.action_space_two_legal_play

    # Test where there isn't anything to play.
    hand = Hand()
    hand.add_cards(cv5, 2)
    hand.add_cards(cv6, 2)
    hand.add_cards(cv7, 2)
    hand.add_cards(cv8, 2)

    action_space = factory.create_play(hand, cv9, 1)
    assert action_space is None

    # Test where there is one card that can be played
    hand = Hand()
    hand.add_cards(cv1, 2)
    hand.add_cards(cv5, 2)
    hand.add_cards(cv6, 2)
    hand.add_cards(cv7, 2)
    hand.add_cards(cv9, 1)

    action_space = factory.create_play(hand, cv8, 1)
    assert action_space == ActionSpaceFactory.action_space_one_legal_play


def test_AllCardActionSpace_Lead():
    """Test AllCardActionSpace when the agent is leading."""

    factory = AllCardActionSpaceFactory()

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

    # Test where there are four cards.
    hand = Hand()
    hand.add_cards(cv1, 1)
    hand.add_cards(cv5, 2)
    hand.add_cards(cv6, 2)
    hand.add_cards(cv7, 1)

    cur_trick_value = None
    cur_trick_count = None
    action_space = factory.create_play(hand, cur_trick_value, cur_trick_count)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 0) == (cv7, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 1) == cv1
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 2) == (cv7, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 3) == (cv7, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 4) == (cv7, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 5) == cv5
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 6) == cv6
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 7) == cv7
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 8) == (cv7, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 9) == (cv7, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 10) == (cv7, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 11) == (cv7, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 12) == (cv7, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 13) == (cv7, -10.0)

    # Test five cards
    hand = Hand()
    hand.add_cards(cv1, 1)
    hand.add_cards(cv5, 2)
    hand.add_cards(cv6, 2)
    hand.add_cards(cv7, 1)
    hand.add_cards(cv9, 1)

    cur_trick_value = None
    cur_trick_count = None
    action_space = factory.create_play(hand, cur_trick_value, cur_trick_count)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 0) == (cv9, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 1) == cv1
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 2) == (cv9, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 3) == (cv9, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 4) == (cv9, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 5) == cv5
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 6) == cv6
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 7) == cv7
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 8) == (cv9, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 9) == cv9
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 10) == (cv9, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 11) == (cv9, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 12) == (cv9, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 13) == (cv9, -10.0)

    # Test when there is an ace in the hand
    hand = Hand()
    hand.add_cards(cv1, 1)
    hand.add_cards(cv5, 2)
    hand.add_cards(cv6, 2)
    hand.add_cards(cv7, 1)
    hand.add_cards(cv9, 1)
    hand.add_cards(cv12, 1)

    cur_trick_value = None
    cur_trick_count = None
    action_space = factory.create_play(hand, cur_trick_value, cur_trick_count)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 0) == (cv12, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 1) == cv1
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 2) == (cv12, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 3) == (cv12, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 4) == (cv12, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 5) == cv5
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 6) == cv6
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 7) == cv7
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 8) == (cv12, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 9) == cv9
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 10) == (cv12, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 11) == (cv12, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 12) == cv12
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 13) == (cv12, -10.0)


def test_AllCardActionSpace_Play():
    """Test AllCardActionSpace when the agent is playing on a trick."""

    factory = AllCardActionSpaceFactory()

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

    # Test where there are four cards and the player should play on an
    # existing single-card trick.
    hand = Hand()
    hand.add_cards(cv1, 1)
    hand.add_cards(cv5, 2)
    hand.add_cards(cv6, 2)
    hand.add_cards(cv7, 1)

    cur_trick_value = cv0
    cur_trick_count = 1
    action_space = factory.create_play(hand, cur_trick_value, cur_trick_count)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 0) == (None, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 1) == cv1
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 2) == (None, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 3) == (None, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 4) == (None, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 5) == cv5
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 6) == cv6
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 7) == cv7
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 8) == (None, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 9) == (None, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 10) == (None, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 11) == (None, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 12) == (None, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 13) == None

    cur_trick_value = cv5
    cur_trick_count = 1
    action_space = factory.create_play(hand, cur_trick_value, cur_trick_count)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 0) == (None, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 1) == (None, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 2) == (None, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 3) == (None, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 4) == (None, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 5) == (None, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 6) == cv6
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 7) == cv7
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 8) == (None, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 9) == (None, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 10) == (None, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 11) == (None, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 12) == (None, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 13) == None

    # Test where there are four cards and the player should play on an
    # ex== double-card trick.
    hand = Hand()
    hand.add_cards(cv1, 1)
    hand.add_cards(cv5, 2)
    hand.add_cards(cv6, 2)
    hand.add_cards(cv7, 1)

    cur_trick_value = cv0
    cur_trick_count = 2
    action_space = factory.create_play(hand, cur_trick_value, cur_trick_count)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 0) == (None, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 1) == (None, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 2) == (None, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 3) == (None, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 4) == (None, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 5) == cv5
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 6) == cv6
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 7) == (None, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 8) == (None, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 9) == (None, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 10) == (None, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 11) == (None, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 12) == (None, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 13) == None

    cur_trick_value = cv5
    cur_trick_count = 2
    action_space = factory.create_play(hand, cur_trick_value, cur_trick_count)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 0) == (None, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 1) == (None, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 2) == (None, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 3) == (None, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 4) == (None, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 5) == (None, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 6) == cv6
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 7) == (None, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 8) == (None, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 9) == (None, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 10) == (None, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 11) == (None, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 12) == (None, -10.0)
    assert action_space.card_to_play(hand, cur_trick_value, cur_trick_count, 13) == None
