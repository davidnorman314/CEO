import pytest
import CEO.CEO.cards.hand as hand

def test_CardValue():
    cv = hand.CardValue(1)
    assert cv.value == 1

    cv = hand.CardValue(0)
    assert cv.value == 0

    cv = hand.CardValue(12)
    assert cv.value == 12

    with pytest.raises(AssertionError):
        cv = hand.CardValue(-1)

    with pytest.raises(AssertionError):
        cv = hand.CardValue(13)

    cv = hand.CardValue(1)
    cv1 = hand.CardValue(1)
    cv2 = hand.CardValue(2)

    assert cv == cv
    assert cv == cv1
    assert cv != cv2
    assert cv1 == cv
    assert cv1 == cv1
    assert cv1 != cv2
    assert cv2 != cv
    assert cv2 != cv1
    assert cv2 == cv2

def test_PlayedCards():
    pc = hand.PlayedCards(hand.CardValue(1), 3)
    assert pc.value == hand.CardValue(1)
    assert pc.count == 3

    with pytest.raises(AssertionError):
        cv = hand.PlayedCards(hand.CardValue(1), 0)

def test_PlayCardsFromHand():
    theHand = hand.Hand()

    theHand.add_cards(hand.CardValue(0), 3)
    pc = hand.PlayedCards(hand.CardValue(0), 1)
    theHand.play_cards(pc)
    assert theHand.to_dict() == {0:2}

    pc = hand.PlayedCards(hand.CardValue(0), 1)
    theHand.play_cards(pc)
    assert theHand.to_dict() == {0:1}

    theHand.add_cards(hand.CardValue(1), 5)
    pc = hand.PlayedCards(hand.CardValue(1), 3)
    theHand.play_cards(pc)
    assert theHand.to_dict() == {0:1, 1:2}

    with pytest.raises(AssertionError):
        pc = hand.PlayedCards(hand.CardValue(0), 2)
        theHand.play_cards(pc)

    with pytest.raises(AssertionError):
        pc = hand.PlayedCards(hand.CardValue(0), 3)
        theHand.play_cards(pc)

    with pytest.raises(AssertionError):
        pc = hand.PlayedCards(hand.CardValue(1), 3)
        theHand.play_cards(pc)

    pc = hand.PlayedCards(hand.CardValue(0), 1)
    theHand.play_cards(pc)
    assert theHand.to_dict() == {1:2}


def test_Hand_cards_equal():
    theHand = hand.Hand()
    theHand.add_cards(hand.CardValue(0), 3)
    theHand.add_cards(hand.CardValue(1), 2)
    theHand.add_cards(hand.CardValue(4), 1)

    assert theHand.cards_equal({0:3, 1:2, 4:1})
    assert not theHand.cards_equal({0:3, 1:2, 4:1, 5:1})
    assert not theHand.cards_equal({0:3, 1:2})
    assert not theHand.cards_equal({0:3, 1:2, 4:2})

    assert theHand.to_dict() == {0:3, 1:2, 4:1}

    theHand.add_cards(hand.CardValue(12), 5)

    assert theHand.to_dict() == {0:3, 1:2, 4:1, 12:5}