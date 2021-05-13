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


def test_DoNotSplitPairs():
    """
    Test that when playing on a trick, the behavior doesn't split up pairs to play on singles,
    except aces can be split.
    """

    # Create CardValue objects for ease of use later
    cv0 = CardValue(0)
    cv1 = CardValue(1)
    cv2 = CardValue(2)
    cv3 = CardValue(3)
    cv4 = CardValue(4)
    cv5 = CardValue(5)
    cv6 = CardValue(6)
    cv11 = CardValue(11)
    cv12 = CardValue(12)

    # Create the object
    behavior = HeuristicBehavior()
    state = RoundState()

    # Test where there are four cards in the hand and there are only players lower than us left to
    # play on the trick.
    hand = Hand()
    hand.add_cards(cv0, 1)
    hand.add_cards(cv1, 2)
    hand.add_cards(cv2, 1)
    hand.add_cards(cv3, 1)

    assert behavior.play_on_trick(0, 3, hand, cv0, 1, state) == cv2
    assert behavior.play_on_trick(0, 3, hand, cv1, 1, state) == cv2
    assert behavior.play_on_trick(0, 3, hand, cv2, 1, state) == cv3
    assert behavior.play_on_trick(0, 3, hand, cv3, 1, state) == None
    assert behavior.play_on_trick(0, 3, hand, cv4, 1, state) == None

    hand = Hand()
    hand.add_cards(cv0, 1)
    hand.add_cards(cv1, 1)
    hand.add_cards(cv2, 2)
    hand.add_cards(cv3, 1)

    assert behavior.play_on_trick(0, 3, hand, cv0, 1, state) == cv1
    assert behavior.play_on_trick(0, 3, hand, cv1, 1, state) == cv3
    assert behavior.play_on_trick(0, 3, hand, cv2, 1, state) == cv3
    assert behavior.play_on_trick(0, 3, hand, cv3, 1, state) == None
    assert behavior.play_on_trick(0, 3, hand, cv4, 1, state) == None

    hand = Hand()
    hand.add_cards(cv0, 1)
    hand.add_cards(cv1, 1)
    hand.add_cards(cv2, 1)
    hand.add_cards(cv3, 2)

    assert behavior.play_on_trick(0, 3, hand, cv0, 1, state) == cv1
    assert behavior.play_on_trick(0, 3, hand, cv1, 1, state) == cv2
    assert behavior.play_on_trick(0, 3, hand, cv2, 1, state) == None
    assert behavior.play_on_trick(0, 3, hand, cv3, 1, state) == None
    assert behavior.play_on_trick(0, 3, hand, cv4, 1, state) == None

    # Test where there are four cards in the hand and there are players higher than us left to
    # play on the trick.
    hand = Hand()
    hand.add_cards(cv0, 1)
    hand.add_cards(cv1, 2)
    hand.add_cards(cv2, 1)
    hand.add_cards(cv3, 1)

    assert behavior.play_on_trick(1, 3, hand, cv0, 1, state) == None
    assert behavior.play_on_trick(1, 3, hand, cv1, 1, state) == None
    assert behavior.play_on_trick(1, 3, hand, cv2, 1, state) == None
    assert behavior.play_on_trick(1, 3, hand, cv3, 1, state) == None
    assert behavior.play_on_trick(1, 3, hand, cv4, 1, state) == None

    hand = Hand()
    hand.add_cards(cv0, 1)
    hand.add_cards(cv1, 1)
    hand.add_cards(cv2, 2)
    hand.add_cards(cv3, 1)

    assert behavior.play_on_trick(1, 3, hand, cv0, 1, state) == cv1
    assert behavior.play_on_trick(1, 3, hand, cv1, 1, state) == None
    assert behavior.play_on_trick(1, 3, hand, cv2, 1, state) == None
    assert behavior.play_on_trick(1, 3, hand, cv3, 1, state) == None
    assert behavior.play_on_trick(1, 3, hand, cv4, 1, state) == None

    hand = Hand()
    hand.add_cards(cv0, 1)
    hand.add_cards(cv1, 1)
    hand.add_cards(cv2, 1)
    hand.add_cards(cv3, 2)

    assert behavior.play_on_trick(1, 3, hand, cv0, 1, state) == cv1
    assert behavior.play_on_trick(1, 3, hand, cv1, 1, state) == None
    assert behavior.play_on_trick(1, 3, hand, cv2, 1, state) == None
    assert behavior.play_on_trick(1, 3, hand, cv3, 1, state) == None
    assert behavior.play_on_trick(1, 3, hand, cv4, 1, state) == None

    # Test where we need to split aces.
    hand = Hand()
    hand.add_cards(cv0, 1)
    hand.add_cards(cv1, 2)
    hand.add_cards(cv2, 1)
    hand.add_cards(cv12, 2)

    assert behavior.play_on_trick(1, 3, hand, cv11, 1, state) == cv12
