import pytest
import CEO.cards.deck as deck
from CEO.cards.hand import *
import CEO.cards.round as rd
import CEO.cards.player as player
from CEO.cards.simplebehavior import *
from CEO.cards.heuristicbehavior import *


def test_LowerVsHigher():
    """
    Test playing cards when there are players higher than us that still have to play.
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
    behavior = HeuristicBehavior()
    state = RoundState()

    # Test where there are four cards in the hand
    hand = Hand()
    hand.add_cards(cv0, 1)
    hand.add_cards(cv1, 1)
    hand.add_cards(cv2, 1)
    hand.add_cards(cv3, 1)

    assert behavior.play_on_trick(2, 3, hand, cv0, 1, state) == cv1
    assert behavior.play_on_trick(2, 3, hand, cv1, 1, state) == None
    assert behavior.play_on_trick(2, 3, hand, cv2, 1, state) == None
    assert behavior.play_on_trick(2, 3, hand, cv3, 1, state) == None
    assert behavior.play_on_trick(2, 3, hand, cv4, 1, state) == None

    # Test where there are five cards in the hand
    hand = Hand()
    hand.add_cards(cv0, 1)
    hand.add_cards(cv1, 1)
    hand.add_cards(cv2, 1)
    hand.add_cards(cv3, 1)
    hand.add_cards(cv4, 1)

    assert behavior.play_on_trick(2, 3, hand, cv0, 1, state) == cv1
    assert behavior.play_on_trick(2, 3, hand, cv1, 1, state) == cv2
    assert behavior.play_on_trick(2, 3, hand, cv2, 1, state) == None
    assert behavior.play_on_trick(2, 3, hand, cv3, 1, state) == None
    assert behavior.play_on_trick(2, 3, hand, cv4, 1, state) == None

    # Test where there are six cards in the hand
    hand = Hand()
    hand.add_cards(cv0, 1)
    hand.add_cards(cv1, 1)
    hand.add_cards(cv2, 1)
    hand.add_cards(cv3, 1)
    hand.add_cards(cv4, 1)
    hand.add_cards(cv5, 1)

    assert behavior.play_on_trick(2, 3, hand, cv0, 1, state) == cv1
    assert behavior.play_on_trick(2, 3, hand, cv1, 1, state) == cv2
    assert behavior.play_on_trick(2, 3, hand, cv2, 1, state) == None
    assert behavior.play_on_trick(2, 3, hand, cv3, 1, state) == None
    assert behavior.play_on_trick(2, 3, hand, cv4, 1, state) == None
    assert behavior.play_on_trick(2, 3, hand, cv5, 1, state) == None
