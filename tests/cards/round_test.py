import ceo.game.eventlistener as el
import ceo.game.hand as hand
import ceo.game.player as player
import ceo.game.round as rd


class StateBase:
    pass


class LeadState(StateBase):
    player_position: int

    def __init__(self, player_position: int):
        self.player_position = player_position

    def __eq__(self, other):
        if isinstance(other, LeadState):
            return self.player_position == other.player_position
        return NotImplemented

    def __str__(self):
        return "(Lead)"


class TrickState(StateBase):
    starting_position: int
    last_player_to_play: int
    player_position: int
    cur_trick_value: hand.CardValue
    cur_trick_count: int

    def __init__(
        self,
        starting_position: int,
        last_player_to_play: int,
        player_position: int,
        cur_trick_value: hand.CardValue,
        cur_trick_count: int,
    ):
        self.starting_position = starting_position
        self.last_player_to_play = last_player_to_play
        self.player_position = player_position
        self.cur_trick_value = cur_trick_value
        self.cur_trick_count = cur_trick_count

    def __eq__(self, other):
        if isinstance(other, TrickState):
            return (
                self.player_position == other.player_position
                and self.cur_trick_value == other.cur_trick_value
                and self.cur_trick_count == other.cur_trick_count
            )
        return NotImplemented

    def __str__(self):
        return (
            "(TrickState "
            + str(self.cur_trick_value)
            + " "
            + str(self.cur_trick_count)
            + ")"
        )

    def __repr__(self):
        return str(self)


class MockPlayerBehavior(player.PlayerBehaviorInterface):
    trick_states: list[TrickState]
    cards_remaining: list[int]

    value_to_play: list[hand.CardValue]
    to_play_next_index: int

    def __init__(self):
        self.value_to_play = []
        self.to_play_next_index = 0
        self.trick_states = []
        self.cards_remaining = []
        self.is_reinforcement_learning = False

    def pass_cards(self, hand: hand.Hand, count: int) -> list[hand.CardValue]:
        raise NotImplementedError

    def lead(
        self, player_position: int, hand: hand.Hand, state: rd.RoundState
    ) -> hand.CardValue:
        self.trick_states.append(LeadState(player_position))
        self.cards_remaining.append(list(state.cards_remaining))

        assert state.last_player_to_play_index is None

        if len(self.value_to_play) <= self.to_play_next_index:
            assert "No more values to play" != ""

        ret = self.value_to_play[self.to_play_next_index]
        self.to_play_next_index += 1

        return ret

    def play_on_trick(
        self,
        starting_position: int,
        player_position: int,
        hand: hand.Hand,
        cur_trick_value: hand.CardValue,
        cur_trick_count: int,
        state: rd.RoundState,
    ) -> hand.CardValue:
        self.trick_states.append(
            TrickState(
                starting_position,
                state.last_player_to_play_index,
                player_position,
                cur_trick_value,
                cur_trick_count,
            )
        )
        self.cards_remaining.append(list(state.cards_remaining))

        if len(self.value_to_play) <= self.to_play_next_index:
            assert "No more values to play" != ""

        ret = self.value_to_play[self.to_play_next_index]
        self.to_play_next_index += 1

        return ret


class MockAsyncBehavior(player.PlayerBehaviorInterface):
    """
    Class used for RL behavior
    """

    to_pass = list()

    def __init__(self):
        self.is_reinforcement_learning = True
        self.to_pass = []

    def pass_cards(self, hand: hand.Hand, count: int) -> list[hand.CardValue]:
        self.to_pass[0]
        self.to_pass.pop(0)

    def lead(
        self, player_position: int, hand: hand.Hand, state: rd.RoundState
    ) -> hand.CardValue:
        assert not "This should not be called"

    def play_on_trick(
        self,
        starting_position: int,
        player_position: int,
        hand: hand.Hand,
        cur_trick_value: hand.CardValue,
        cur_trick_count: int,
        state: rd.RoundState,
    ) -> hand.CardValue:
        assert not "This should not be called"


def test_simpleround():
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
    hand2.add_cards(cv2, 1)

    hand3 = hand.Hand()
    hand3.add_cards(cv2, 1)
    hand3.add_cards(cv5, 2)
    hand3.add_cards(cv0, 1)

    hand4 = hand.Hand()
    hand4.add_cards(cv3, 1)
    hand4.add_cards(cv2, 2)
    hand4.add_cards(cv1, 1)

    # Make the players
    behavior1 = MockPlayerBehavior()
    behavior2 = MockPlayerBehavior()
    behavior3 = MockPlayerBehavior()
    behavior4 = MockPlayerBehavior()

    behavior1.value_to_play.append(cv0)
    behavior2.value_to_play.append(cv1)
    behavior3.value_to_play.append(cv2)
    behavior4.value_to_play.append(cv3)

    behavior4.value_to_play.append(cv2)
    behavior1.value_to_play.append(cv3)
    behavior2.value_to_play.append(cv4)
    behavior3.value_to_play.append(cv5)

    behavior3.value_to_play.append(cv0)
    behavior4.value_to_play.append(cv1)
    behavior2.value_to_play.append(cv2)

    player1 = player.Player("Player1", behavior1)
    player2 = player.Player("Player2", behavior2)
    player3 = player.Player("Player3", behavior3)
    player4 = player.Player("Player4", behavior4)

    # Play the round
    listener = el.PrintAllEventListener()
    round = rd.Round(
        [player1, player2, player3, player4], [hand1, hand2, hand3, hand4], listener
    )
    round.play()

    # Check that the behavior objects were correctly called
    assert behavior1.trick_states[0] == LeadState(0)
    assert behavior2.trick_states[0] == TrickState(0, 1, 1, cv0, 1)
    assert behavior3.trick_states[0] == TrickState(0, 2, 2, cv1, 1)
    assert behavior4.trick_states[0] == TrickState(0, 3, 3, cv2, 1)

    assert behavior4.trick_states[1] == LeadState(3)
    assert behavior1.trick_states[1] == TrickState(3, 3, 0, cv2, 2)
    assert behavior2.trick_states[1] == TrickState(3, 0, 1, cv3, 2)
    assert behavior3.trick_states[1] == TrickState(3, 1, 2, cv4, 2)

    assert behavior3.trick_states[2] == LeadState(2)
    assert behavior4.trick_states[2] == TrickState(2, 2, 3, cv0, 1)
    assert behavior2.trick_states[2] == TrickState(2, 3, 1, cv1, 1)

    assert len(behavior1.trick_states) == 2
    assert len(behavior2.trick_states) == 3
    assert len(behavior3.trick_states) == 3
    assert len(behavior4.trick_states) == 3

    # All hands should be empty
    assert hand1.to_dict() == {}
    assert hand2.to_dict() == {}
    assert hand3.to_dict() == {}
    assert hand4.to_dict() == {}

    # Check the next round odder
    assert round.get_next_round_order() == [0, 2, 3, 1]
    assert round.get_final_ceo_card_count() == 0

    # Check cards_remaining
    cards_remaining = behavior1.cards_remaining
    assert cards_remaining[0] == [3, 4, 4, 4]
    assert cards_remaining[1] == [2, 3, 3, 1]
    assert len(cards_remaining) == 2

    cards_remaining = behavior2.cards_remaining
    assert cards_remaining[0] == [2, 4, 4, 4]
    assert cards_remaining[1] == [0, 3, 3, 1]
    assert cards_remaining[2] == [0, 1, 0, 0]
    assert len(cards_remaining) == 3

    cards_remaining = behavior3.cards_remaining
    assert cards_remaining[0] == [2, 3, 4, 4]
    assert cards_remaining[1] == [0, 1, 3, 1]
    assert cards_remaining[2] == [0, 1, 1, 1]
    assert len(cards_remaining) == 3

    cards_remaining = behavior4.cards_remaining
    assert cards_remaining[0] == [2, 3, 3, 4]
    assert cards_remaining[1] == [2, 3, 3, 3]
    assert cards_remaining[2] == [0, 1, 0, 1]
    assert len(cards_remaining) == 3


def test_passing():
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
    hand2.add_cards(cv0, 1)

    hand3 = hand.Hand()
    hand3.add_cards(cv2, 1)
    hand3.add_cards(cv2, 2)
    hand3.add_cards(cv1, 1)

    hand4 = hand.Hand()
    hand4.add_cards(cv3, 2)
    hand4.add_cards(cv2, 1)

    # Make the players
    behavior1 = MockPlayerBehavior()
    behavior2 = MockPlayerBehavior()
    behavior3 = MockPlayerBehavior()
    behavior4 = MockPlayerBehavior()

    behavior1.value_to_play.append(cv0)
    behavior2.value_to_play.append(cv1)
    behavior3.value_to_play.append(cv2)
    behavior4.value_to_play.append(None)

    behavior3.value_to_play.append(cv2)
    behavior4.value_to_play.append(cv3)
    behavior1.value_to_play.append(cv4)
    behavior2.value_to_play.append(cv5)

    behavior2.value_to_play.append(cv0)
    behavior3.value_to_play.append(cv1)
    behavior4.value_to_play.append(cv2)

    player1 = player.Player("Player1", behavior1)
    player2 = player.Player("Player2", behavior2)
    player3 = player.Player("Player3", behavior3)
    player4 = player.Player("Player4", behavior4)

    # Play the round
    listener = el.PrintAllEventListener()
    round = rd.Round(
        [player1, player2, player3, player4], [hand1, hand2, hand3, hand4], listener
    )
    round.play()

    # Check that the behavior objects were correctly called
    assert behavior1.trick_states[0] == LeadState(0)
    assert behavior2.trick_states[0] == TrickState(0, 0, 1, cv0, 1)
    assert behavior3.trick_states[0] == TrickState(0, 1, 2, cv1, 1)
    assert behavior4.trick_states[0] == TrickState(0, 2, 3, cv2, 1)

    assert behavior3.trick_states[1] == LeadState(2)
    assert behavior4.trick_states[1] == TrickState(2, 2, 3, cv2, 2)
    assert behavior1.trick_states[1] == TrickState(2, 3, 0, cv3, 2)
    assert behavior2.trick_states[1] == TrickState(2, 0, 1, cv4, 2)

    assert behavior2.trick_states[2] == LeadState(1)
    assert behavior3.trick_states[2] == TrickState(1, 1, 2, cv0, 1)
    assert behavior4.trick_states[2] == TrickState(1, 2, 3, cv1, 1)

    assert len(behavior1.trick_states) == 2
    assert len(behavior2.trick_states) == 3
    assert len(behavior3.trick_states) == 3
    assert len(behavior4.trick_states) == 3

    # All hands should be empty
    assert hand1.to_dict() == {}
    assert hand2.to_dict() == {}
    assert hand3.to_dict() == {}
    assert hand4.to_dict() == {}

    # Check the next round order
    assert round.get_next_round_order() == [0, 1, 2, 3]
    assert round.get_final_ceo_card_count() == 0


def test_skipemptyhand():
    """
    Test playing a round of CEO where a player doesn't have cards
    """

    # Create CardValue objects for ease of use later
    cv0 = hand.CardValue(0)
    cv1 = hand.CardValue(1)
    cv2 = hand.CardValue(2)
    hand.CardValue(3)
    hand.CardValue(4)
    hand.CardValue(5)

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
    round = rd.Round(
        [player1, player2, player3, player4], [hand1, hand2, hand3, hand4], listener
    )
    round.play()

    # Check that the behavior objects were correctly called
    assert behavior1.trick_states[0] == LeadState(0)
    assert behavior2.trick_states[0] == TrickState(0, 0, 1, cv0, 1)
    assert behavior4.trick_states[0] == TrickState(0, 1, 3, cv1, 1)

    assert len(behavior1.trick_states) == 1
    assert len(behavior2.trick_states) == 1
    assert len(behavior3.trick_states) == 0
    assert len(behavior4.trick_states) == 1

    # All hands should be empty
    assert hand1.to_dict() == {}
    assert hand2.to_dict() == {}
    assert hand3.to_dict() == {}
    assert hand4.to_dict() == {}


def test_leadafterplayergoesoutnotceo():
    """
    Test that the game correctly determines who leads after a
    player goes out as the last player who plays on a trick.
    Here CEO is still in when the player goes out, so they
    stop playing and go to the bottom.
    """

    # Create CardValue objects for ease of use later
    cv0 = hand.CardValue(0)
    cv1 = hand.CardValue(1)
    cv2 = hand.CardValue(2)
    cv3 = hand.CardValue(3)
    cv4 = hand.CardValue(4)
    hand.CardValue(5)

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

    behavior2.value_to_play.append(cv3)
    behavior4.value_to_play.append(cv4)

    player1 = player.Player("Player1", behavior1)
    player2 = player.Player("Player2", behavior2)
    player3 = player.Player("Player3", behavior3)
    player4 = player.Player("Player4", behavior4)

    # Play the round
    listener = el.PrintAllEventListener()
    round = rd.Round(
        [player1, player2, player3, player4], [hand1, hand2, hand3, hand4], listener
    )
    round.play()

    # Check that the behavior objects were correctly called
    assert behavior1.trick_states[0] == LeadState(0)
    assert behavior2.trick_states[0] == TrickState(0, 0, 1, cv0, 1)
    assert behavior3.trick_states[0] == TrickState(0, 1, 2, cv1, 1)
    assert behavior4.trick_states[0] == TrickState(0, 2, 3, cv2, 1)

    assert behavior2.trick_states[1] == LeadState(1)
    assert behavior4.trick_states[1] == TrickState(1, 1, 3, cv3, 2)

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
    assert round.get_next_round_order() == [2, 1, 3, 0]
    assert round.get_final_ceo_card_count() == 2


def test_leadafterplayergoesoutceo():
    """
    Test that the game correctly determines who leads after a
    player goes out as the last player who plays on a trick.
    Here CEO is is the one who goes out.
    """

    # Create CardValue objects for ease of use later
    hand.CardValue(0)
    cv1 = hand.CardValue(1)
    cv2 = hand.CardValue(2)
    cv3 = hand.CardValue(3)
    hand.CardValue(4)
    cv5 = hand.CardValue(5)

    # Make the hands
    hand1 = hand.Hand()
    hand1.add_cards(cv5, 1)

    hand2 = hand.Hand()
    hand2.add_cards(cv1, 1)

    hand3 = hand.Hand()
    hand3.add_cards(cv2, 1)

    hand4 = hand.Hand()
    hand4.add_cards(cv3, 1)

    # Make the players
    behavior1 = MockPlayerBehavior()
    behavior2 = MockPlayerBehavior()
    behavior3 = MockPlayerBehavior()
    behavior4 = MockPlayerBehavior()

    behavior1.value_to_play.append(cv5)
    behavior2.value_to_play.append(None)
    behavior3.value_to_play.append(None)
    behavior4.value_to_play.append(None)

    behavior2.value_to_play.append(cv1)
    behavior3.value_to_play.append(cv2)
    behavior4.value_to_play.append(cv3)

    player1 = player.Player("Player1", behavior1)
    player2 = player.Player("Player2", behavior2)
    player3 = player.Player("Player3", behavior3)
    player4 = player.Player("Player4", behavior4)

    # Play the round
    listener = el.PrintAllEventListener()
    round = rd.Round(
        [player1, player2, player3, player4], [hand1, hand2, hand3, hand4], listener
    )
    round.play()

    # Check that the behavior objects were correctly called
    assert behavior1.trick_states[0] == LeadState(0)
    assert behavior2.trick_states[0] == TrickState(0, 0, 1, cv5, 1)
    assert behavior3.trick_states[0] == TrickState(0, 0, 2, cv5, 1)
    assert behavior4.trick_states[0] == TrickState(0, 0, 3, cv5, 1)

    assert behavior2.trick_states[1] == LeadState(1)
    assert behavior3.trick_states[1] == TrickState(1, 1, 2, cv1, 1)
    assert behavior4.trick_states[1] == TrickState(1, 2, 3, cv2, 1)

    assert len(behavior1.trick_states) == 1
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
    assert round.get_final_ceo_card_count() == 0


def test_leadafterplayergoesout2():
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
    hand.CardValue(5)

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
    round = rd.Round(
        [player1, player2, player3, player4], [hand1, hand2, hand3, hand4], listener
    )
    round.play()

    # Check that the behavior objects were correctly called
    assert behavior1.trick_states[0] == LeadState(0)
    assert behavior2.trick_states[0] == TrickState(0, 0, 1, cv0, 1)
    assert behavior3.trick_states[0] == TrickState(0, 1, 2, cv1, 1)
    assert behavior4.trick_states[0] == TrickState(0, 2, 3, cv2, 1)

    assert behavior2.trick_states[1] == LeadState(1)
    assert behavior4.trick_states[1] == TrickState(1, 2, 3, cv3, 2)

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
    assert round.get_final_ceo_card_count() == 0


def test_nooneplaysontrick():
    """
    Test where a player leads and then no one else plays on the trick
    """

    # Create CardValue objects for ease of use later
    cv0 = hand.CardValue(0)
    cv1 = hand.CardValue(1)
    cv2 = hand.CardValue(2)
    hand.CardValue(3)
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
    round = rd.Round(
        [player1, player2, player3, player4], [hand1, hand2, hand3, hand4], listener
    )
    round.play()

    # Check that the behavior objects were correctly called
    assert behavior1.trick_states[0] == LeadState(0)
    assert behavior2.trick_states[0] == TrickState(0, 0, 1, cv5, 1)
    assert behavior3.trick_states[0] == TrickState(0, 0, 2, cv5, 1)
    assert behavior4.trick_states[0] == TrickState(0, 0, 3, cv5, 1)

    assert behavior1.trick_states[1] == LeadState(0)
    assert behavior2.trick_states[1] == TrickState(0, 1, 1, cv0, 1)
    assert behavior3.trick_states[1] == TrickState(0, 2, 2, cv1, 1)
    assert behavior4.trick_states[1] == TrickState(0, 3, 3, cv2, 1)

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
    assert round.get_final_ceo_card_count() == 0


def test_ceodoesnotgooutfirstmiddle():
    """
    Test when someone goes out before the CEO.
    Here the player goes out in the middle of a trick
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
    hand3.add_cards(cv5, 1)

    hand4 = hand.Hand()
    hand4.add_cards(cv3, 1)
    hand4.add_cards(cv4, 1)

    # Make the players
    behavior1 = MockPlayerBehavior()
    behavior2 = MockPlayerBehavior()
    behavior3 = MockPlayerBehavior()
    behavior4 = MockPlayerBehavior()

    behavior1.value_to_play.append(cv0)
    behavior2.value_to_play.append(cv1)
    behavior3.value_to_play.append(cv2)
    behavior4.value_to_play.append(cv3)

    behavior4.value_to_play.append(cv4)
    behavior3.value_to_play.append(cv5)

    player1 = player.Player("Player1", behavior1)
    player2 = player.Player("Player2", behavior2)
    player3 = player.Player("Player3", behavior3)
    player4 = player.Player("Player4", behavior4)

    # Play the round
    listener = el.PrintAllEventListener()
    round = rd.Round(
        [player1, player2, player3, player4], [hand1, hand2, hand3, hand4], listener
    )
    round.play()

    # Check that the behavior objects were correctly called
    assert behavior1.trick_states[0] == LeadState(0)
    assert behavior2.trick_states[0] == TrickState(0, 1, 1, cv0, 1)
    assert behavior3.trick_states[0] == TrickState(0, 2, 2, cv1, 1)
    assert behavior4.trick_states[0] == TrickState(0, 3, 3, cv2, 1)

    assert behavior4.trick_states[1] == LeadState(3)
    assert behavior3.trick_states[1] == TrickState(3, 3, 2, cv4, 1)

    assert len(behavior1.trick_states) == 1
    assert len(behavior2.trick_states) == 1
    assert len(behavior3.trick_states) == 2
    assert len(behavior4.trick_states) == 2

    # All hands should be empty
    assert hand1.to_dict() == {}
    assert hand2.to_dict() == {}
    assert hand3.to_dict() == {}
    assert hand4.to_dict() == {}

    # Check the next round order
    assert round.get_next_round_order() == [1, 3, 2, 0]
    assert round.get_final_ceo_card_count() == 1


def test_ceodoesnotgooutfirst_lead():
    """
    Test when someone goes out before the CEO.
    Here the player goes out by leading
    """

    # Create CardValue objects for ease of use later
    cv0 = hand.CardValue(0)
    cv1 = hand.CardValue(1)
    cv2 = hand.CardValue(2)
    cv3 = hand.CardValue(3)
    cv4 = hand.CardValue(4)
    cv5 = hand.CardValue(5)
    cv6 = hand.CardValue(6)

    # Make the hands
    hand1 = hand.Hand()
    hand1.add_cards(cv0, 1)
    hand1.add_cards(cv5, 1)

    hand2 = hand.Hand()
    hand2.add_cards(cv1, 1)
    hand2.add_cards(cv5, 1)
    hand2.add_cards(cv1, 1)

    hand3 = hand.Hand()
    hand3.add_cards(cv2, 1)
    hand3.add_cards(cv6, 1)
    hand3.add_cards(cv0, 1)

    hand4 = hand.Hand()
    hand4.add_cards(cv3, 1)
    hand4.add_cards(cv4, 1)

    # Make the players
    behavior1 = MockPlayerBehavior()
    behavior2 = MockPlayerBehavior()
    behavior3 = MockPlayerBehavior()
    behavior4 = MockPlayerBehavior()

    behavior1.value_to_play.append(cv0)
    behavior2.value_to_play.append(cv1)
    behavior3.value_to_play.append(cv2)
    behavior4.value_to_play.append(cv3)

    behavior4.value_to_play.append(cv4)
    behavior2.value_to_play.append(cv5)
    behavior3.value_to_play.append(cv6)

    behavior3.value_to_play.append(cv0)
    behavior2.value_to_play.append(cv1)

    player1 = player.Player("Player1", behavior1)
    player2 = player.Player("Player2", behavior2)
    player3 = player.Player("Player3", behavior3)
    player4 = player.Player("Player4", behavior4)

    # Play the round
    listener = el.PrintAllEventListener()
    round = rd.Round(
        [player1, player2, player3, player4], [hand1, hand2, hand3, hand4], listener
    )
    round.play()

    # Check that the behavior objects were correctly called
    assert behavior1.trick_states[0] == LeadState(0)
    assert behavior2.trick_states[0] == TrickState(0, 1, 1, cv0, 1)
    assert behavior3.trick_states[0] == TrickState(0, 2, 2, cv1, 1)
    assert behavior4.trick_states[0] == TrickState(0, 3, 3, cv2, 1)

    assert behavior4.trick_states[1] == LeadState(3)
    assert behavior2.trick_states[1] == TrickState(3, 3, 1, cv4, 1)
    assert behavior3.trick_states[1] == TrickState(3, 1, 2, cv5, 1)

    assert behavior3.trick_states[2] == LeadState(2)
    assert behavior2.trick_states[2] == TrickState(2, 2, 1, cv0, 1)

    assert len(behavior1.trick_states) == 1
    assert len(behavior2.trick_states) == 3
    assert len(behavior3.trick_states) == 3
    assert len(behavior4.trick_states) == 2

    # All hands should be empty
    assert hand1.to_dict() == {}
    assert hand2.to_dict() == {}
    assert hand3.to_dict() == {}
    assert hand4.to_dict() == {}

    # Check the next round order
    assert round.get_next_round_order() == [3, 2, 1, 0]
    assert round.get_final_ceo_card_count() == 1


def test_asyncround():
    """
    Test playing a quick round of CEO using the generator interface for a player
    behavior.
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
    hand2.add_cards(cv2, 1)

    hand3 = hand.Hand()
    hand3.add_cards(cv2, 1)
    hand3.add_cards(cv5, 2)
    hand3.add_cards(cv0, 1)

    hand4 = hand.Hand()
    hand4.add_cards(cv3, 1)
    hand4.add_cards(cv2, 2)
    hand4.add_cards(cv1, 1)

    # Make the players
    behavior1 = MockPlayerBehavior()
    behavior2 = MockPlayerBehavior()
    behavior3 = MockPlayerBehavior()

    to_lead4 = []
    to_play4 = []

    behavior1.value_to_play.append(cv0)
    behavior2.value_to_play.append(cv1)
    behavior3.value_to_play.append(cv2)
    to_play4.append(cv3)

    to_lead4.append(cv2)
    behavior1.value_to_play.append(cv3)
    behavior2.value_to_play.append(cv4)
    behavior3.value_to_play.append(cv5)

    behavior3.value_to_play.append(cv0)
    to_play4.append(cv1)
    behavior2.value_to_play.append(cv2)

    player4_trick_states = []
    player4_cards_remaining = []

    player1 = player.Player("Player1", behavior1)
    player2 = player.Player("Player2", behavior2)
    player3 = player.Player("Player3", behavior3)
    player4 = player.Player("Player4", MockAsyncBehavior())

    # Play the round. Player 4 is asynchronous
    listener = el.PrintAllEventListener()
    round = rd.Round(
        [player1, player2, player3, player4], [hand1, hand2, hand3, hand4], listener
    )
    gen = round.play_generator()

    try:
        gen_tuple = next(gen)

        while True:
            if gen_tuple[0] == "lead":
                player4_trick_states.append(LeadState(gen_tuple[1]))
                player4_cards_remaining.append(list(gen_tuple[3].cards_remaining))
                cv = to_lead4.pop(0)
            elif gen_tuple[0] == "play":
                player4_trick_states.append(
                    TrickState(
                        gen_tuple[1],
                        gen_tuple[6].last_player_to_play_index,
                        gen_tuple[2],
                        gen_tuple[4],
                        gen_tuple[5],
                    )
                )
                player4_cards_remaining.append(list(gen_tuple[6].cards_remaining))
                cv = to_play4.pop(0)
            else:
                assert "Unexpected action" == ""

            print("Playing", cv)
            gen_tuple = gen.send(cv)
    except StopIteration:
        print("Iteration finished")

    # Check that the behavior objects were correctly called
    assert behavior1.trick_states[0] == LeadState(0)
    assert behavior2.trick_states[0] == TrickState(0, 1, 1, cv0, 1)
    assert behavior3.trick_states[0] == TrickState(0, 2, 2, cv1, 1)
    assert player4_trick_states[0] == TrickState(0, 3, 3, cv2, 1)

    assert player4_trick_states[1] == LeadState(3)
    assert behavior1.trick_states[1] == TrickState(3, 3, 0, cv2, 2)
    assert behavior2.trick_states[1] == TrickState(3, 0, 1, cv3, 2)
    assert behavior3.trick_states[1] == TrickState(3, 1, 2, cv4, 2)

    assert behavior3.trick_states[2] == LeadState(2)
    assert player4_trick_states[2] == TrickState(2, 2, 3, cv0, 1)
    assert behavior2.trick_states[2] == TrickState(2, 3, 1, cv1, 1)

    assert len(behavior1.trick_states) == 2
    assert len(behavior2.trick_states) == 3
    assert len(behavior3.trick_states) == 3
    assert len(player4_trick_states) == 3

    # All hands should be empty
    assert hand1.to_dict() == {}
    assert hand2.to_dict() == {}
    assert hand3.to_dict() == {}
    assert hand4.to_dict() == {}

    # Check the next round odder
    assert round.get_next_round_order() == [0, 2, 3, 1]

    # Check cards_remaining
    cards_remaining = behavior1.cards_remaining
    assert cards_remaining[0] == [3, 4, 4, 4]
    assert cards_remaining[1] == [2, 3, 3, 1]
    assert len(cards_remaining) == 2

    cards_remaining = behavior2.cards_remaining
    assert cards_remaining[0] == [2, 4, 4, 4]
    assert cards_remaining[1] == [0, 3, 3, 1]
    assert cards_remaining[2] == [0, 1, 0, 0]
    assert len(cards_remaining) == 3

    cards_remaining = behavior3.cards_remaining
    assert cards_remaining[0] == [2, 3, 4, 4]
    assert cards_remaining[1] == [0, 1, 3, 1]
    assert cards_remaining[2] == [0, 1, 1, 1]
    assert len(cards_remaining) == 3

    cards_remaining = player4_cards_remaining
    assert cards_remaining[0] == [2, 3, 3, 4]
    assert cards_remaining[1] == [2, 3, 3, 3]
    assert cards_remaining[2] == [0, 1, 0, 1]
    assert len(cards_remaining) == 3
