from CEO.cards.hand import *
from CEO.cards.player import *
from CEO.cards.eventlistener import *


class Round:
    """
    Class representing on round in a game of CEO
    """

    def __init__(self, players: list[Player], hands: list[Hand], listener: EventListenerInterface):
        self._players = players
        self._hands = hands
        self._player_count = len(self._players)
        self._next_round_order = []
        self._listener = listener

        assert len(self._players) == len(self._hands)

    def play(self):
        starting_player = 0
        trick_number = 0
        while not self._all_cards_played():
            trick_number += 1
            assert trick_number < len(self._hands) * 13

            starting_player = self._play_trick(starting_player)

    def _play_trick(self, starting_player: int) -> int:
        # print("Staring trick with player number ", starting_player)
        state = RoundState()

        # Calculate the order of the other players after the player that leads.
        # Note that play_order is an iterator.
        play_order = map(
            (lambda x: (x + starting_player) % self._player_count), range(self._player_count)
        )
        next(play_order)

        # Handle the lead
        cur_player = self._players[starting_player]
        cur_hand = self._hands[starting_player]

        assert not cur_hand.is_empty()

        cur_card_value = cur_player.behavoir.lead(cur_hand, state)
        cur_card_count = cur_hand.count(cur_card_value)

        # print(starting_player, " ", cur_player, " leads ", cur_card_value)
        self._listener.lead(cur_card_value, cur_card_count, starting_player, cur_player)

        self._play_cards(starting_player, cur_card_value, cur_card_count)

        assert cur_card_value is not None
        assert cur_card_count > 0

        # Let the remaining players play
        last_index_to_play = -1
        for cur_index in play_order:
            cur_player = self._players[cur_index]
            cur_hand = self._hands[cur_index]

            if cur_hand.is_empty():
                continue

            new_card_value = cur_player.behavoir.play_on_trick(
                cur_hand, cur_card_value, cur_card_count, state
            )

            if new_card_value is None:
                # print(cur_index, " ", cur_player, " passes")
                self._listener.pass_on_trick(cur_index, cur_player)
                continue

            self._play_cards(cur_index, new_card_value, cur_card_count)

            # print(cur_index, " ", cur_player, " plays ", new_card_value)
            self._listener.play_cards(new_card_value, cur_card_count, cur_index, cur_player)

            cur_card_value = new_card_value
            last_index_to_play = cur_index

        # If the last player to play on the trick went out, then
        # find the next player to lead.
        if self._hands[last_index_to_play].is_empty():
            last_index_to_play = 0
            while (
                last_index_to_play < self._player_count
                and self._hands[last_index_to_play].is_empty()
            ):
                last_index_to_play += 1

        return last_index_to_play

    def get_next_round_order(self) -> list[int]:
        return self._next_round_order

    def _play_cards(self, player_index: int, card_value: CardValue, count: int):
        theHand = self._hands[player_index]
        theHand.remove_cards(card_value, count)

        if theHand.is_empty():
            self._next_round_order.append(player_index)

    def _all_cards_played(self):
        return all(map((lambda x: x.is_empty()), self._hands))
