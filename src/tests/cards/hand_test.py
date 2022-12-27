import pytest
import CEO.cards.hand as hand


def test_CardValue():
    cv = hand.CardValue(1)
    assert cv.value == 1

    cv = hand.CardValue(0)
    assert cv.value == 0

    cv = hand.CardValue(12)
    assert cv.value == 12

    with pytest.raises(ValueError):
        cv = hand.CardValue(-1)

    with pytest.raises(ValueError):
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
    assert theHand.to_dict() == {0: 2}

    pc = hand.PlayedCards(hand.CardValue(0), 1)
    theHand.play_cards(pc)
    assert theHand.to_dict() == {0: 1}

    theHand.add_cards(hand.CardValue(1), 5)
    pc = hand.PlayedCards(hand.CardValue(1), 3)
    theHand.play_cards(pc)
    assert theHand.to_dict() == {0: 1, 1: 2}

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
    assert theHand.to_dict() == {1: 2}


def test_GetCardValues():
    theHand = hand.Hand()

    cv0 = hand.CardValue(0)
    cv3 = hand.CardValue(3)
    cv5 = hand.CardValue(5)
    cv7 = hand.CardValue(7)

    theHand.add_cards(cv0, 3)
    assert theHand.get_card_values() == [(cv0, 3)]

    theHand.add_cards(cv3, 5)
    assert theHand.get_card_values() == [(cv0, 3), (cv3, 5)]

    theHand.add_cards(cv5, 2)
    assert theHand.get_card_values() == [(cv0, 3), (cv3, 5), (cv5, 2)]


def test_HandCardCount():
    theHand = hand.Hand()

    cv0 = hand.CardValue(0)
    cv3 = hand.CardValue(3)

    theHand.add_cards(cv0, 3)
    theHand.add_cards(cv3, 5)

    assert theHand.count(cv0) == 3
    assert theHand.count(cv3) == 5


def test_MaxCardValue():
    theHand = hand.Hand()

    cv0 = hand.CardValue(0)
    cv3 = hand.CardValue(3)
    cv11 = hand.CardValue(11)
    cv12 = hand.CardValue(12)

    theHand.add_cards(cv0, 3)
    theHand.add_cards(cv3, 5)
    assert theHand.max_card_value() == cv3

    theHand.add_cards(cv11, 2)
    assert theHand.max_card_value() == cv11

    theHand.add_cards(cv12, 2)
    assert theHand.max_card_value() == cv12


def test_Hand_cards_equal():
    theHand = hand.Hand()
    theHand.add_cards(hand.CardValue(0), 3)
    theHand.add_cards(hand.CardValue(1), 2)
    theHand.add_cards(hand.CardValue(4), 1)

    assert theHand.cards_equal({0: 3, 1: 2, 4: 1})
    assert not theHand.cards_equal({0: 3, 1: 2, 4: 1, 5: 1})
    assert not theHand.cards_equal({0: 3, 1: 2})
    assert not theHand.cards_equal({0: 3, 1: 2, 4: 2})

    assert theHand.to_dict() == {0: 3, 1: 2, 4: 1}

    theHand.add_cards(hand.CardValue(12), 5)

    assert theHand.to_dict() == {0: 3, 1: 2, 4: 1, 12: 5}
