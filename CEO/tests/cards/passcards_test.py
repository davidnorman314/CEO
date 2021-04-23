import pytest
import CEO.CEO.cards.deck as deck
import CEO.CEO.cards.hand as hand
import CEO.CEO.cards.player as player
import CEO.CEO.cards.passcards as pc
import CEO.CEO.cards.round as rd
import CEO.CEO.cards.eventlistener as el

class MockPlayerBehavior(player.PlayerBehaviorInterface):
    to_pass : list[hand.CardValue]
    pass_called : bool
    
    def __init__(self):
        self.to_pass = None
        self.pass_called = False

    def pass_cards(self, hand: hand.Hand, count: int) -> list[deck.CardValue]:
        self.pass_called = True
        return self.to_pass

    def lead(self, hand: hand.Hand, state: rd.RoundState) -> hand.CardValue:
        pass

    def playOnTrick(self, hand: hand.Hand, cur_trick_value : hand.CardValue, 
                cur_trick_count: int, state: rd.RoundState) -> hand.CardValue:
        pass

def test_FivePlayers():
    """
    Test passing cards with five players
    """

    # Create CardValue objects for ease of use later
    cv0 = hand.CardValue(0)
    cv1 = hand.CardValue(1)
    cv2 = hand.CardValue(2)
    cv3 = hand.CardValue(3)
    cv4 = hand.CardValue(4)
    cv5 = hand.CardValue(5)

    # Make the hands
    hand1 = hand.Hand()
    hand1.add_cards(cv0, 1)
    hand1.add_cards(cv1, 1)
    hand1.add_cards(cv2, 1)
    hand1.add_cards(cv3, 2)

    hand2 = hand.Hand()
    hand2.add_cards(cv1, 1)
    hand2.add_cards(cv4, 2)

    hand3 = hand.Hand()
    hand3.add_cards(cv2, 1)

    hand4 = hand.Hand()
    hand4.add_cards(cv2, 2)
    hand4.add_cards(cv3, 1)
    hand4.add_cards(cv4, 2)

    hand5 = hand.Hand()
    hand5.add_cards(cv3, 1)
    hand5.add_cards(cv5, 2)

    # Make the players
    behavior1 = MockPlayerBehavior()
    behavior1.to_pass = [cv0, cv1]

    behavior2 = MockPlayerBehavior()
    behavior2.to_pass = [cv1]

    behavior3 = MockPlayerBehavior()

    behavior4 = MockPlayerBehavior()

    behavior5 = MockPlayerBehavior()
    
    player1 = player.Player("Player1", behavior1)
    player2 = player.Player("Player2", behavior2)
    player3 = player.Player("Player3", behavior3)
    player4 = player.Player("Player4", behavior4)
    player5 = player.Player("Player5", behavior5)

    # Do passing
    listener = el.PrintAllEventListener()
    passcards = pc.PassCards([player1, player2, player3, player4, player5], 
                             [hand1, hand2, hand3, hand4, hand5],
                             listener)
    passcards.do_card_passing()

    # Check that the behavior objects were correctly called
    assert behavior1.pass_called
    assert behavior2.pass_called
    assert not behavior3.pass_called
    assert not behavior4.pass_called
    assert not behavior5.pass_called

    # Check the hands
    assert hand1.to_dict() == { 2 : 1, 3 : 2, 5 : 2}
    assert hand2.to_dict() == { 4 : 3}
    assert hand3.to_dict() == { 2 : 1}
    assert hand4.to_dict() == { 1 : 1, 2 : 2, 3 : 1, 4 : 1}
    assert hand5.to_dict() == { 0 : 1, 1 : 1, 3 : 1}

def test_FourPlayers():
    """
    Test passing cards with four players
    """

    # Create CardValue objects for ease of use later
    cv0 = hand.CardValue(0)
    cv1 = hand.CardValue(1)
    cv2 = hand.CardValue(2)
    cv3 = hand.CardValue(3)
    cv4 = hand.CardValue(4)
    cv5 = hand.CardValue(5)

    # Make the hands
    hand1 = hand.Hand()
    hand1.add_cards(cv0, 1)
    hand1.add_cards(cv1, 1)
    hand1.add_cards(cv2, 1)
    hand1.add_cards(cv3, 2)

    hand2 = hand.Hand()
    hand2.add_cards(cv1, 1)
    hand2.add_cards(cv4, 2)

    hand3 = hand.Hand()
    hand3.add_cards(cv2, 2)
    hand3.add_cards(cv3, 1)
    hand3.add_cards(cv4, 2)

    hand4 = hand.Hand()
    hand4.add_cards(cv3, 1)
    hand4.add_cards(cv4, 1)
    hand4.add_cards(cv5, 1)

    # Make the players
    behavior1 = MockPlayerBehavior()
    behavior1.to_pass = [cv0, cv1]

    behavior2 = MockPlayerBehavior()
    behavior2.to_pass = [cv1]

    behavior3 = MockPlayerBehavior()

    behavior4 = MockPlayerBehavior()
    
    player1 = player.Player("Player1", behavior1)
    player2 = player.Player("Player2", behavior2)
    player3 = player.Player("Player3", behavior3)
    player4 = player.Player("Player4", behavior4)

    # Do passing
    listener = el.PrintAllEventListener()
    passcards = pc.PassCards([player1, player2, player3, player4], 
                             [hand1, hand2, hand3, hand4],
                             listener)
    passcards.do_card_passing()

    # Check that the behavior objects were correctly called
    assert behavior1.pass_called
    assert behavior2.pass_called
    assert not behavior3.pass_called
    assert not behavior4.pass_called

    # Check the hands
    assert hand1.to_dict() == { 2 : 1, 3 : 2, 4 : 1, 5 : 1}
    assert hand2.to_dict() == { 4 : 3}
    assert hand3.to_dict() == { 1 : 1, 2 : 2, 3 : 1, 4 : 1}
    assert hand4.to_dict() == { 0 : 1, 1 : 1, 3 : 1}