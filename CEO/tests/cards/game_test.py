import pytest
import CEO.cards.round as rd
import CEO.cards.player as player
import CEO.cards.game as g
import CEO.cards.eventlistener as el

from unittest.mock import MagicMock, Mock, patch, call


def players_to_name_list(players: list[player.Player]):
    return list(map(lambda player: player.name, players))


@patch("CEO.cards.game.random")
@patch("CEO.cards.game.Round")
def test_game_player_order(MockRoundClass, MockRandom):
    """
    Test that players have the correct order from round to round.
    """

    # Mock random.shuffle so that it doesn't do anything. This means we don't
    # randomly assign players to seats at the start of the round.
    MockRandom.shuffe.return_value = None

    # Set up the round results
    instance = MockRoundClass.return_value
    instance.get_next_round_order.side_effect = [[3, 2, 1, 0], [1, 2, 3, 0], [0, 1, 2, 3]]

    # Make the players
    player1 = player.Player("Player1", None)
    player2 = player.Player("Player2", None)
    player3 = player.Player("Player3", None)
    player4 = player.Player("Player4", None)

    listener = el.PrintAllEventListener()

    game = g.Game([player1, player2, player3, player4], listener)
    game.play(3)

    # Check that the rounds were created with the correct players.
    # We need to filter the call list to get the constructor calls
    mock_calls = MockRoundClass.mock_calls
    ctr_calls = list(filter(lambda call: call[0] == "", mock_calls))

    index = 0
    ctr_call_args = ctr_calls[index][1]
    players = players_to_name_list(ctr_call_args[0])
    assert players == ["Player1", "Player2", "Player3", "Player4"]

    index += 1
    ctr_call_args = ctr_calls[index][1]
    players = players_to_name_list(ctr_call_args[0])
    assert players == ["Player4", "Player3", "Player2", "Player1"]

    index += 1
    ctr_call_args = ctr_calls[index][1]
    players = players_to_name_list(ctr_call_args[0])
    assert players == ["Player3", "Player2", "Player1", "Player4"]
