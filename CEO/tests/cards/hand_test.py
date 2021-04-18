import pytest
import CEO.CEO.cards.hand as hand

def test_add():
    theHand = hand.Hand()

    assert theHand.add(1,2) == 3