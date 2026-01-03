import pytest

import CEO.cards.hand as hand


def test_cardvalue():
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


def test_playedcards():
    pc = hand.PlayedCards(hand.CardValue(1), 3)
    assert pc.value == hand.CardValue(1)
    assert pc.count == 3

    with pytest.raises(AssertionError):
        hand.PlayedCards(hand.CardValue(1), 0)


def test_playcardsfromhand():
    the_hand = hand.Hand()

    the_hand.add_cards(hand.CardValue(0), 3)
    pc = hand.PlayedCards(hand.CardValue(0), 1)
    the_hand.play_cards(pc)
    assert the_hand.to_dict() == {0: 2}

    pc = hand.PlayedCards(hand.CardValue(0), 1)
    the_hand.play_cards(pc)
    assert the_hand.to_dict() == {0: 1}

    the_hand.add_cards(hand.CardValue(1), 5)
    pc = hand.PlayedCards(hand.CardValue(1), 3)
    the_hand.play_cards(pc)
    assert the_hand.to_dict() == {0: 1, 1: 2}

    with pytest.raises(AssertionError):
        pc = hand.PlayedCards(hand.CardValue(0), 2)
        the_hand.play_cards(pc)

    with pytest.raises(AssertionError):
        pc = hand.PlayedCards(hand.CardValue(0), 3)
        the_hand.play_cards(pc)

    with pytest.raises(AssertionError):
        pc = hand.PlayedCards(hand.CardValue(1), 3)
        the_hand.play_cards(pc)

    pc = hand.PlayedCards(hand.CardValue(0), 1)
    the_hand.play_cards(pc)
    assert the_hand.to_dict() == {1: 2}


def test_getcardvalues():
    the_hand = hand.Hand()

    cv0 = hand.CardValue(0)
    cv3 = hand.CardValue(3)
    cv5 = hand.CardValue(5)
    hand.CardValue(7)

    the_hand.add_cards(cv0, 3)
    assert the_hand.get_card_values() == [(cv0, 3)]

    the_hand.add_cards(cv3, 5)
    assert the_hand.get_card_values() == [(cv0, 3), (cv3, 5)]

    the_hand.add_cards(cv5, 2)
    assert the_hand.get_card_values() == [(cv0, 3), (cv3, 5), (cv5, 2)]


def test_handcardcount():
    the_hand = hand.Hand()

    cv0 = hand.CardValue(0)
    cv3 = hand.CardValue(3)

    the_hand.add_cards(cv0, 3)
    the_hand.add_cards(cv3, 5)

    assert the_hand.count(cv0) == 3
    assert the_hand.count(cv3) == 5


def test_maxcardvalue():
    the_hand = hand.Hand()

    cv0 = hand.CardValue(0)
    cv3 = hand.CardValue(3)
    cv11 = hand.CardValue(11)
    cv12 = hand.CardValue(12)

    the_hand.add_cards(cv0, 3)
    the_hand.add_cards(cv3, 5)
    assert the_hand.max_card_value() == cv3

    the_hand.add_cards(cv11, 2)
    assert the_hand.max_card_value() == cv11

    the_hand.add_cards(cv12, 2)
    assert the_hand.max_card_value() == cv12


def test_handcardsequal():
    the_hand = hand.Hand()
    the_hand.add_cards(hand.CardValue(0), 3)
    the_hand.add_cards(hand.CardValue(1), 2)
    the_hand.add_cards(hand.CardValue(4), 1)

    assert the_hand.cards_equal({0: 3, 1: 2, 4: 1})
    assert not the_hand.cards_equal({0: 3, 1: 2, 4: 1, 5: 1})
    assert not the_hand.cards_equal({0: 3, 1: 2})
    assert not the_hand.cards_equal({0: 3, 1: 2, 4: 2})

    assert the_hand.to_dict() == {0: 3, 1: 2, 4: 1}

    the_hand.add_cards(hand.CardValue(12), 5)

    assert the_hand.to_dict() == {0: 3, 1: 2, 4: 1, 12: 5}
