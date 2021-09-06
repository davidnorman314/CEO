import pytest
import CEO.cards.deck as deck
from CEO.cards.hand import *
import CEO.cards.round as rd
import CEO.cards.player as player
from gym_ceo.envs.actions import Actions


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
