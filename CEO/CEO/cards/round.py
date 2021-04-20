from CEO.CEO.cards.hand import *
from CEO.CEO.cards.player import *

class Round:
    """
    Class representing on round in a game of CEO
    """
    def __init__(self, players : list[Player], hands : list[Hand]):
        self._players = players
        self._hands = hands
        self._player_count = len(self._players)

        assert len(self._players) == len(self._hands)

    def play(self):
        print("Staring round")

        starting_player = 0
        trick_number = 0
        while not self._all_cards_played():
            trick_number += 1
            assert trick_number < len(self._hands) * 13

            starting_player = self._play_trick(starting_player)

    def _play_trick(self, starting_player : int) -> int:
        print("Staring trick with player number ", starting_player)
        state = RoundState()

        # Calculate the order of the other players after the player that leads.
        # Note that play_order is an iterator.
        play_order = map( (lambda x : (x + starting_player) % self._player_count), 
                            range(self._player_count))
        next(play_order)
        
        # Handle the lead
        cur_player = self._players[starting_player]
        cur_hand = self._hands[starting_player]

        cur_card_value = cur_player.behavoir.lead(cur_hand, state)
        cur_card_count = cur_hand.count(cur_card_value)
        cur_hand.remove_cards(cur_card_value, cur_card_count)

        print(starting_player, " ", cur_player, " leads ", cur_card_value)

        assert cur_card_value is not None
        assert cur_card_count > 0

        # Let the remaining players play
        last_index_to_play = -1
        for cur_index in play_order:
            cur_player = self._players[cur_index]
            cur_hand = self._hands[cur_index]

            if cur_hand.is_empty():
                continue

            new_card_value = cur_player.behavoir.playOnTrick(
                cur_hand, cur_card_value, cur_card_count, state)

            if new_card_value is None:
                print(cur_index, " ", cur_player, " passes")
                continue

            print(cur_index, " ", cur_player, " plays ", new_card_value)

            assert new_card_value is not None
            cur_hand.remove_cards(new_card_value, cur_card_count)

            cur_card_value = new_card_value
            last_index_to_play = cur_index

        return last_index_to_play

    def _all_cards_played(self):
        return all(map((lambda x : x.is_empty()), self._hands))
