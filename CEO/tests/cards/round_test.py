import pytest
import CEO.CEO.cards.deck as deck
import CEO.CEO.cards.hand as hand
import CEO.CEO.cards.round as rd
import CEO.CEO.cards.player as player

class StateBase:
    pass

class LeadState(StateBase):
    def __init__(self):
        pass

    def __eq__(self, other):
        if isinstance(other, LeadState):
            return True
        return NotImplemented

    def __str__(self):
        return "(Lead)"

class TrickState(StateBase):
    cur_trick_value: hand.CardValue
    cur_trick_count: int

    def __init__(self, cur_trick_value : hand.CardValue, cur_trick_count : int):
        self.cur_trick_value = cur_trick_value 
        self.cur_trick_count = cur_trick_count

    def __eq__(self, other):
        if isinstance(other, TrickState):
            return self.cur_trick_value == other.cur_trick_value and self.cur_trick_count == other.cur_trick_count
        return NotImplemented

    def __str__(self):
        return "(TrickState " + str(self.cur_trick_value) + " " + str(self.cur_trick_count) + ")"

    def __repr__(self):
        return str(self)

class MockPlayerBehavior(player.PlayerBehaviorInterface):
    trick_states: list[TrickState]
    
    value_to_play: list[hand.CardValue]
    to_play_next_index: int

    def __init__(self):
        self.value_to_play = []
        self.to_play_next_index = 0
        self.trick_states = []

    def lead(self, hand: hand.Hand, state: rd.RoundState) -> hand.CardValue:
        self.trick_states.append(LeadState())

        ret = self.value_to_play[self.to_play_next_index]
        self.to_play_next_index += 1

        return ret

    def playOnTrick(self, hand: hand.Hand, cur_trick_value : hand.CardValue, 
                cur_trick_count: int, state: rd.RoundState) -> hand.CardValue:
        self.trick_states.append(TrickState(cur_trick_value, cur_trick_count))

        ret = self.value_to_play[self.to_play_next_index]
        self.to_play_next_index += 1

        return ret

def test_SimpleRound():
    """
    Test playing a quick round of CEO
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
    hand1.add_cards(cv3, 2)

    hand2 = hand.Hand()
    hand2.add_cards(cv1, 1)
    hand2.add_cards(cv4, 2)

    hand3 = hand.Hand()
    hand3.add_cards(cv2, 1)
    hand3.add_cards(cv5, 2)

    hand4 = hand.Hand()
    hand4.add_cards(cv3, 1)
    hand4.add_cards(cv2, 2)

    # Make the players
    behavior1 = MockPlayerBehavior()
    behavior1.value_to_play.append(cv0)
    behavior1.value_to_play.append(cv3)

    behavior2 = MockPlayerBehavior()
    behavior2.value_to_play.append(cv1)
    behavior2.value_to_play.append(cv4)

    behavior3 = MockPlayerBehavior()
    behavior3.value_to_play.append(cv2)
    behavior3.value_to_play.append(cv5)

    behavior4 = MockPlayerBehavior()
    behavior4.value_to_play.append(cv3)
    behavior4.value_to_play.append(cv2)
    
    player1 = player.Player("Player1", behavior1)
    player2 = player.Player("Player2", behavior2)
    player3 = player.Player("Player3", behavior3)
    player4 = player.Player("Player4", behavior4)

    # Play the round
    round = rd.Round([player1, player2, player3, player4], 
                     [hand1, hand2, hand3, hand4])
    round.play()

    # Check that the behavior objects were correctly called
    assert behavior1.trick_states[0] == LeadState()
    assert behavior2.trick_states[0] == TrickState(cv0, 1)
    assert behavior3.trick_states[0] == TrickState(cv1, 1)
    assert behavior4.trick_states[0] == TrickState(cv2, 1)

    assert behavior4.trick_states[1] == LeadState()
    assert behavior1.trick_states[1] == TrickState(cv2, 2)
    assert behavior2.trick_states[1] == TrickState(cv3, 2)
    assert behavior3.trick_states[1] == TrickState(cv4, 2)

    assert len(behavior1.trick_states) == 2
    assert len(behavior2.trick_states) == 2
    assert len(behavior3.trick_states) == 2
    assert len(behavior4.trick_states) == 2

    # All hands should be empty
    assert hand1.to_dict() == {}
    assert hand2.to_dict() == {}
    assert hand3.to_dict() == {}
    assert hand4.to_dict() == {}