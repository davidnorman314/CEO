import pytest
import CEO.cards.deck as deck
import CEO.cards.hand as hand
import CEO.cards.round as rd
import CEO.cards.player as player
import CEO.cards.eventlistener as el


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

    def __init__(self, cur_trick_value: hand.CardValue, cur_trick_count: int):
        self.cur_trick_value = cur_trick_value
        self.cur_trick_count = cur_trick_count

    def __eq__(self, other):
        if isinstance(other, TrickState):
            return (
                self.cur_trick_value == other.cur_trick_value
                and self.cur_trick_count == other.cur_trick_count
            )
        return NotImplemented

    def __str__(self):
        return "(TrickState " + str(self.cur_trick_value) + " " + str(self.cur_trick_count) + ")"

    def __repr__(self):
        return str(self)


class MockPlayerBehavior(player.PlayerBehaviorInterface):
    trick_states: list[TrickState]
    cards_remaining: list[rd.RoundState]

    value_to_play: list[hand.CardValue]
    to_play_next_index: int

    def __init__(self):
        self.value_to_play = []
        self.to_play_next_index = 0
        self.trick_states = []
        self.cards_remaining = []

    def lead(self, hand: hand.Hand, state: rd.RoundState) -> hand.CardValue:
        self.trick_states.append(LeadState())
        self.cards_remaining.append(state.cards_remaining)

        ret = self.value_to_play[self.to_play_next_index]
        self.to_play_next_index += 1

        return ret

    def play_on_trick(
        self,
        hand: hand.Hand,
        cur_trick_value: hand.CardValue,
        cur_trick_count: int,
        state: rd.RoundState,
    ) -> hand.CardValue:
        self.trick_states.append(TrickState(cur_trick_value, cur_trick_count))
        self.cards_remaining.append(state.cards_remaining)

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
    listener = el.PrintAllEventListener()
    round = rd.Round([player1, player2, player3, player4], [hand1, hand2, hand3, hand4], listener)
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

    # Check the next round odder
    assert round.get_next_round_order() == [3, 0, 1, 2]

    # Check cards_remaining
    cards_remaining = behavior1.cards_remaining
    assert cards_remaining[0] == [3, 3, 3, 3]
    assert cards_remaining[1] == [2, 2, 2, 0]
    assert len(cards_remaining) == 2

    cards_remaining = behavior2.cards_remaining
    assert cards_remaining[0] == [2, 3, 3, 3]
    assert cards_remaining[1] == [0, 2, 2, 0]
    assert len(cards_remaining) == 2

    cards_remaining = behavior3.cards_remaining
    assert cards_remaining[0] == [2, 2, 3, 3]
    assert cards_remaining[1] == [0, 0, 2, 0]
    assert len(cards_remaining) == 2

    cards_remaining = behavior4.cards_remaining
    assert cards_remaining[0] == [2, 2, 2, 3]
    assert cards_remaining[1] == [2, 2, 2, 2]
    assert len(cards_remaining) == 2


def test_Passing():
    """
    Test playing a round of CEO with passing
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
    hand1.add_cards(cv4, 2)

    hand2 = hand.Hand()
    hand2.add_cards(cv1, 1)
    hand2.add_cards(cv5, 2)

    hand3 = hand.Hand()
    hand3.add_cards(cv2, 1)
    hand3.add_cards(cv2, 2)

    hand4 = hand.Hand()
    hand4.add_cards(cv3, 2)

    # Make the players
    behavior1 = MockPlayerBehavior()
    behavior1.value_to_play.append(cv0)
    behavior1.value_to_play.append(cv4)

    behavior2 = MockPlayerBehavior()
    behavior2.value_to_play.append(cv1)
    behavior2.value_to_play.append(cv5)

    behavior3 = MockPlayerBehavior()
    behavior3.value_to_play.append(cv2)
    behavior3.value_to_play.append(cv2)

    behavior4 = MockPlayerBehavior()
    behavior4.value_to_play.append(None)
    behavior4.value_to_play.append(cv3)

    player1 = player.Player("Player1", behavior1)
    player2 = player.Player("Player2", behavior2)
    player3 = player.Player("Player3", behavior3)
    player4 = player.Player("Player4", behavior4)

    # Play the round
    listener = el.PrintAllEventListener()
    round = rd.Round([player1, player2, player3, player4], [hand1, hand2, hand3, hand4], listener)
    round.play()

    # Check that the behavior objects were correctly called
    assert behavior1.trick_states[0] == LeadState()
    assert behavior2.trick_states[0] == TrickState(cv0, 1)
    assert behavior3.trick_states[0] == TrickState(cv1, 1)
    assert behavior4.trick_states[0] == TrickState(cv2, 1)

    assert behavior3.trick_states[1] == LeadState()
    assert behavior4.trick_states[1] == TrickState(cv2, 2)
    assert behavior1.trick_states[1] == TrickState(cv3, 2)
    assert behavior2.trick_states[1] == TrickState(cv4, 2)

    assert len(behavior1.trick_states) == 2
    assert len(behavior2.trick_states) == 2
    assert len(behavior3.trick_states) == 2
    assert len(behavior4.trick_states) == 2

    # All hands should be empty
    assert hand1.to_dict() == {}
    assert hand2.to_dict() == {}
    assert hand3.to_dict() == {}
    assert hand4.to_dict() == {}

    # Check the next round order
    assert round.get_next_round_order() == [2, 3, 0, 1]


def test_SkipEmptyHand():
    """
    Test playing a round of CEO where a player doesn't have cards
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

    hand2 = hand.Hand()
    hand2.add_cards(cv1, 1)

    hand3 = hand.Hand()

    hand4 = hand.Hand()
    hand4.add_cards(cv2, 1)

    # Make the players
    behavior1 = MockPlayerBehavior()
    behavior1.value_to_play.append(cv0)

    behavior2 = MockPlayerBehavior()
    behavior2.value_to_play.append(cv1)

    behavior3 = MockPlayerBehavior()

    behavior4 = MockPlayerBehavior()
    behavior4.value_to_play.append(cv2)

    player1 = player.Player("Player1", behavior1)
    player2 = player.Player("Player2", behavior2)
    player3 = player.Player("Player3", behavior3)
    player4 = player.Player("Player4", behavior4)

    # Play the round
    listener = el.PrintAllEventListener()
    round = rd.Round([player1, player2, player3, player4], [hand1, hand2, hand3, hand4], listener)
    round.play()

    # Check that the behavior objects were correctly called
    assert behavior1.trick_states[0] == LeadState()
    assert behavior2.trick_states[0] == TrickState(cv0, 1)
    assert behavior4.trick_states[0] == TrickState(cv1, 1)

    assert len(behavior1.trick_states) == 1
    assert len(behavior2.trick_states) == 1
    assert len(behavior3.trick_states) == 0
    assert len(behavior4.trick_states) == 1

    # All hands should be empty
    assert hand1.to_dict() == {}
    assert hand2.to_dict() == {}
    assert hand3.to_dict() == {}
    assert hand4.to_dict() == {}


def test_LeadAfterPlayerGoesOut():
    """
    Test that the game correctly determines who leads after a
    player goes out as the last player who plays on a trick.
    Here the next player to lead is CEO, e.g., index zero.
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
    hand1.add_cards(cv2, 2)

    hand2 = hand.Hand()
    hand2.add_cards(cv1, 1)
    hand2.add_cards(cv3, 2)

    hand3 = hand.Hand()
    hand3.add_cards(cv2, 1)

    hand4 = hand.Hand()
    hand4.add_cards(cv4, 2)

    # Make the players
    behavior1 = MockPlayerBehavior()
    behavior2 = MockPlayerBehavior()
    behavior3 = MockPlayerBehavior()
    behavior4 = MockPlayerBehavior()

    behavior1.value_to_play.append(cv0)
    behavior2.value_to_play.append(cv1)
    behavior3.value_to_play.append(cv2)
    behavior4.value_to_play.append(None)

    behavior1.value_to_play.append(cv2)
    behavior2.value_to_play.append(cv3)
    behavior4.value_to_play.append(cv4)

    player1 = player.Player("Player1", behavior1)
    player2 = player.Player("Player2", behavior2)
    player3 = player.Player("Player3", behavior3)
    player4 = player.Player("Player4", behavior4)

    # Play the round
    listener = el.PrintAllEventListener()
    round = rd.Round([player1, player2, player3, player4], [hand1, hand2, hand3, hand4], listener)
    round.play()

    # Check that the behavior objects were correctly called
    assert behavior1.trick_states[0] == LeadState()
    assert behavior2.trick_states[0] == TrickState(cv0, 1)
    assert behavior3.trick_states[0] == TrickState(cv1, 1)
    assert behavior4.trick_states[0] == TrickState(cv2, 1)

    assert behavior1.trick_states[1] == LeadState()
    assert behavior2.trick_states[1] == TrickState(cv2, 2)
    assert behavior4.trick_states[1] == TrickState(cv3, 2)

    assert len(behavior1.trick_states) == 2
    assert len(behavior2.trick_states) == 2
    assert len(behavior3.trick_states) == 1
    assert len(behavior4.trick_states) == 2

    # All hands should be empty
    assert hand1.to_dict() == {}
    assert hand2.to_dict() == {}
    assert hand3.to_dict() == {}
    assert hand4.to_dict() == {}

    # Check the next round order
    assert round.get_next_round_order() == [2, 0, 1, 3]


def test_LeadAfterPlayerGoesOut2():
    """
    Test that the game correctly determines who leads after a
    player goes out as the last player who plays on a trick.
    Here the next player to lead is the second player, since CEO is out.
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

    hand2 = hand.Hand()
    hand2.add_cards(cv1, 1)
    hand2.add_cards(cv3, 2)

    hand3 = hand.Hand()
    hand3.add_cards(cv2, 1)

    hand4 = hand.Hand()
    hand4.add_cards(cv4, 2)

    # Make the players
    behavior1 = MockPlayerBehavior()
    behavior2 = MockPlayerBehavior()
    behavior3 = MockPlayerBehavior()
    behavior4 = MockPlayerBehavior()

    behavior1.value_to_play.append(cv0)
    behavior2.value_to_play.append(cv1)
    behavior3.value_to_play.append(cv2)
    behavior4.value_to_play.append(None)

    behavior2.value_to_play.append(cv3)
    behavior4.value_to_play.append(cv4)

    player1 = player.Player("Player1", behavior1)
    player2 = player.Player("Player2", behavior2)
    player3 = player.Player("Player3", behavior3)
    player4 = player.Player("Player4", behavior4)

    # Play the round
    listener = el.PrintAllEventListener()
    round = rd.Round([player1, player2, player3, player4], [hand1, hand2, hand3, hand4], listener)
    round.play()

    # Check that the behavior objects were correctly called
    assert behavior1.trick_states[0] == LeadState()
    assert behavior2.trick_states[0] == TrickState(cv0, 1)
    assert behavior3.trick_states[0] == TrickState(cv1, 1)
    assert behavior4.trick_states[0] == TrickState(cv2, 1)

    assert behavior2.trick_states[1] == LeadState()
    assert behavior4.trick_states[1] == TrickState(cv3, 2)

    assert len(behavior1.trick_states) == 1
    assert len(behavior2.trick_states) == 2
    assert len(behavior3.trick_states) == 1
    assert len(behavior4.trick_states) == 2

    # All hands should be empty
    assert hand1.to_dict() == {}
    assert hand2.to_dict() == {}
    assert hand3.to_dict() == {}
    assert hand4.to_dict() == {}

    # Check the next round order
    assert round.get_next_round_order() == [0, 2, 1, 3]


def test_NoOnePlaysOnTrick():
    """
    Test where a player leads and then no one else plays on the trick
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
    hand1.add_cards(cv5, 1)

    hand2 = hand.Hand()
    hand2.add_cards(cv1, 1)

    hand3 = hand.Hand()
    hand3.add_cards(cv2, 1)

    hand4 = hand.Hand()
    hand4.add_cards(cv4, 1)

    # Make the players
    behavior1 = MockPlayerBehavior()
    behavior2 = MockPlayerBehavior()
    behavior3 = MockPlayerBehavior()
    behavior4 = MockPlayerBehavior()

    behavior1.value_to_play.append(cv5)
    behavior2.value_to_play.append(None)
    behavior3.value_to_play.append(None)
    behavior4.value_to_play.append(None)

    behavior1.value_to_play.append(cv0)
    behavior2.value_to_play.append(cv1)
    behavior3.value_to_play.append(cv2)
    behavior4.value_to_play.append(cv4)

    player1 = player.Player("Player1", behavior1)
    player2 = player.Player("Player2", behavior2)
    player3 = player.Player("Player3", behavior3)
    player4 = player.Player("Player4", behavior4)

    # Play the round
    listener = el.PrintAllEventListener()
    round = rd.Round([player1, player2, player3, player4], [hand1, hand2, hand3, hand4], listener)
    round.play()

    # Check that the behavior objects were correctly called
    assert behavior1.trick_states[0] == LeadState()
    assert behavior2.trick_states[0] == TrickState(cv5, 1)
    assert behavior3.trick_states[0] == TrickState(cv5, 1)
    assert behavior4.trick_states[0] == TrickState(cv5, 1)

    assert behavior1.trick_states[1] == LeadState()
    assert behavior2.trick_states[1] == TrickState(cv0, 1)
    assert behavior3.trick_states[1] == TrickState(cv1, 1)
    assert behavior4.trick_states[1] == TrickState(cv2, 1)

    assert len(behavior1.trick_states) == 2
    assert len(behavior2.trick_states) == 2
    assert len(behavior3.trick_states) == 2
    assert len(behavior4.trick_states) == 2

    # All hands should be empty
    assert hand1.to_dict() == {}
    assert hand2.to_dict() == {}
    assert hand3.to_dict() == {}
    assert hand4.to_dict() == {}

    # Check the next round order
    assert round.get_next_round_order() == [0, 1, 2, 3]
