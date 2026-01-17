import ceo.game.deck as deck
import ceo.game.hand as hand


def test_deal():
    the_deck = deck.Deck(4)

    hands = the_deck.deal()

    # Check that we have four hands
    assert len(hands) == 4

    # Check that each hand has 13 cards
    for the_hand in hands:
        cards = the_hand.to_dict()

        card_counts = [cards[key] for key in cards]

        assert sum(card_counts) == 13

    print(hands[0])
    print(hands[1])
    print(hands[2])
    print(hands[3])

    # Check that each rank was dealt correctly
    for value in range(13):
        card_value = hand.CardValue(value)
        counts = [the_hand.count(card_value) for the_hand in hands]

        assert sum(counts) == 4
