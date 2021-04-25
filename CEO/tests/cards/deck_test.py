import pytest
import CEO.cards.deck as deck
import CEO.cards.hand as hand


def test_Deal():

    theDeck = deck.Deck(4)

    hands = theDeck.deal()

    # Check that we have four hands
    assert len(hands) == 4

    # Check that each hand has 13 cards
    for theHand in hands:
        cards = theHand.to_dict()

        cardCounts = [cards[key] for key in cards]

        assert sum(cardCounts) == 13

    print(hands[0])
    print(hands[1])
    print(hands[2])
    print(hands[3])

    # Check that each rank was dealt correctly
    for value in range(13):
        card_value = hand.CardValue(value)
        counts = [theHand.count(card_value) for theHand in hands]

        assert sum(counts) == 4
