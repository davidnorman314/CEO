import pytest
import CEO.CEO.cards.deck as deck
from CEO.CEO.cards.hand import *
import CEO.CEO.cards.round as rd
import CEO.CEO.cards.player as player
from CEO.CEO.cards.simplebehavior import *


def test_SimpleBehaviorBase_pass_singles():
    """
    Test the pass_singles method
    """

    # Create CardValue objects for ease of use later
    cv0 = CardValue(0)
    cv1 = CardValue(1)
    cv2 = CardValue(2)
    cv3 = CardValue(3)
    cv4 = CardValue(4)
    cv5 = CardValue(5)
    cv6 = CardValue(6)

    # Create the object
    behavior = SimpleBehaviorBase()

    # Test passing the two lowest cards
    hand = Hand()
    hand.add_cards(cv0, 1)
    hand.add_cards(cv1, 1)
    hand.add_cards(cv2, 1)
    hand.add_cards(cv3, 2)

    assert behavior.pass_singles(hand, 2) == [cv0, cv1]

    hand = Hand()
    hand.add_cards(cv0, 1)
    hand.add_cards(cv2, 1)
    hand.add_cards(cv3, 1)
    hand.add_cards(cv4, 2)

    assert behavior.pass_singles(hand, 2) == [cv0, cv2]

    # Test passing the three lowest cards
    hand = Hand()
    hand.add_cards(cv0, 1)
    hand.add_cards(cv1, 1)
    hand.add_cards(cv2, 1)
    hand.add_cards(cv3, 2)

    assert behavior.pass_singles(hand, 3) == [cv0, cv1, cv2]

    hand = Hand()
    hand.add_cards(cv1, 1)
    hand.add_cards(cv3, 1)
    hand.add_cards(cv5, 1)
    hand.add_cards(cv6, 2)

    assert behavior.pass_singles(hand, 3) == [cv1, cv3, cv5]

    # Test skipping a double in the middle of passed cards
    hand = Hand()
    hand.add_cards(cv0, 1)
    hand.add_cards(cv1, 2)
    hand.add_cards(cv2, 1)
    hand.add_cards(cv3, 2)

    assert behavior.pass_singles(hand, 2) == [cv0, cv2]

    # Test skipping a double as the lowest card
    hand = Hand()
    hand.add_cards(cv0, 2)
    hand.add_cards(cv1, 1)
    hand.add_cards(cv2, 1)
    hand.add_cards(cv3, 2)

    assert behavior.pass_singles(hand, 2) == [cv1, cv2]

    # Test where there aren't enough singles
    hand = Hand()
    hand.add_cards(cv0, 2)
    hand.add_cards(cv1, 1)
    hand.add_cards(cv2, 1)
    hand.add_cards(cv3, 2)

    assert behavior.pass_singles(hand, 3) == [cv0, cv0, cv1]

    hand = Hand()
    hand.add_cards(cv0, 2)
    hand.add_cards(cv3, 1)
    hand.add_cards(cv4, 1)
    hand.add_cards(cv5, 2)

    assert behavior.pass_singles(hand, 3) == [cv0, cv0, cv3]

    hand = Hand()
    hand.add_cards(cv0, 1)
    hand.add_cards(cv1, 2)
    hand.add_cards(cv2, 1)
    hand.add_cards(cv3, 2)

    assert behavior.pass_singles(hand, 3) == [cv0, cv1, cv1]

    hand = Hand()
    hand.add_cards(cv0, 1)
    hand.add_cards(cv1, 2)
    hand.add_cards(cv2, 2)
    hand.add_cards(cv3, 1)

    assert behavior.pass_singles(hand, 3) == [cv0, cv1, cv1]

    hand = Hand()
    hand.add_cards(cv0, 1)
    hand.add_cards(cv1, 2)
    hand.add_cards(cv2, 2)
    hand.add_cards(cv3, 1)

    assert behavior.pass_singles(hand, 3) == [cv0, cv1, cv1]

    hand = Hand()
    hand.add_cards(cv0, 2)
    hand.add_cards(cv1, 2)
    hand.add_cards(cv2, 2)
    hand.add_cards(cv3, 1)

    assert behavior.pass_singles(hand, 3) == [cv0, cv0, cv3]

    # Test where there aren't enough singles and there is a triple
    hand = Hand()
    hand.add_cards(cv0, 1)
    hand.add_cards(cv1, 3)
    hand.add_cards(cv2, 1)
    hand.add_cards(cv3, 2)

    assert behavior.pass_singles(hand, 3) == [cv0, cv2, cv3]


def test_SimpleBehaviorBase_play_lowest_or_pass():
    """
    Test the play_lowest_or_pass method
    """

    # Create CardValue objects for ease of use later
    cv0 = CardValue(0)
    cv1 = CardValue(1)
    cv2 = CardValue(2)
    cv3 = CardValue(3)
    cv4 = CardValue(4)
    cv5 = CardValue(5)
    cv6 = CardValue(6)

    # Create the object
    behavior = SimpleBehaviorBase()
    state = RoundState()

    # Test playing a single
    hand = Hand()
    hand.add_cards(cv0, 1)
    hand.add_cards(cv1, 1)
    hand.add_cards(cv2, 1)
    hand.add_cards(cv3, 2)
    hand.add_cards(cv4, 1)

    assert behavior.play_lowest_or_pass(hand, cv0, 1, state) == cv1
    assert behavior.play_lowest_or_pass(hand, cv1, 1, state) == cv2
    assert behavior.play_lowest_or_pass(hand, cv2, 1, state) == cv3
    assert behavior.play_lowest_or_pass(hand, cv3, 1, state) == cv4
    assert behavior.play_lowest_or_pass(hand, cv4, 1, state) == None
    assert behavior.play_lowest_or_pass(hand, cv5, 1, state) == None

    # Test playing a pair
    hand = Hand()
    hand.add_cards(cv0, 1)
    hand.add_cards(cv1, 2)
    hand.add_cards(cv2, 1)
    hand.add_cards(cv3, 2)
    hand.add_cards(cv4, 1)

    assert behavior.play_lowest_or_pass(hand, cv0, 2, state) == cv1
    assert behavior.play_lowest_or_pass(hand, cv1, 2, state) == cv3
    assert behavior.play_lowest_or_pass(hand, cv2, 2, state) == cv3
    assert behavior.play_lowest_or_pass(hand, cv3, 2, state) == None
    assert behavior.play_lowest_or_pass(hand, cv4, 2, state) == None
    assert behavior.play_lowest_or_pass(hand, cv5, 2, state) == None


def test_SimpleBehaviorBase_lead_lowest():
    """
    Test the lead_lowest method
    """

    # Create CardValue objects for ease of use later
    cv0 = CardValue(0)
    cv1 = CardValue(1)
    cv2 = CardValue(2)
    cv3 = CardValue(3)
    cv4 = CardValue(4)
    cv5 = CardValue(5)
    cv6 = CardValue(6)

    # Create the object
    behavior = SimpleBehaviorBase()
    state = RoundState()

    # Test
    hand = Hand()
    hand.add_cards(cv0, 1)
    hand.add_cards(cv1, 1)
    hand.add_cards(cv2, 1)

    assert behavior.lead_lowest(hand, state) == cv0

    hand = Hand()
    hand.add_cards(cv1, 1)
    hand.add_cards(cv2, 1)

    assert behavior.lead_lowest(hand, state) == cv1

    hand = Hand()
    hand.add_cards(cv4, 1)
    hand.add_cards(cv5, 1)

    assert behavior.lead_lowest(hand, state) == cv4
