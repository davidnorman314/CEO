from CEO.cards.player import *
from CEO.cards.hand import *
from CEO.cards.eventlistener import *
from CEO.cards.passcards import *
from CEO.cards.round import *
from CEO.cards.eventlistener import *

import random


class Game:
    """
    Class that plays a multiple-round game of CEO.
    """

    _seats: list[int]
    _listener: EventListenerInterface
    _players: list[Player]
    _player_count: int

    def __init__(self, players: list[Player], listener: EventListenerInterface):
        self._listener = listener
        self._players = players
        self._player_count = len(players)

    def play(self, *, round_count: int = 500, do_shuffle: bool = True):
        """
        Play a game with a given number of rounds
        """

        self._seats = list(range(len(self._players)))

        if do_shuffle:
            random.shuffle(self._seats)

        for i in range(round_count):
            players_for_round = [self._players[i] for i in self._seats]

            if i > 0:
                self._listener.end_round(players_for_round)

            self._listener.start_round(players_for_round)

            deck = Deck(self._player_count)
            hands = deck.deal()

            passcards = PassCards(players_for_round, hands, self._listener)
            passcards.do_card_passing()

            round = Round(players_for_round, hands, self._listener)
            round.play()

            next_round_order = round.get_next_round_order()
            assert len(self._seats) == len(next_round_order)

            self._seats = list(map(lambda i: self._seats[i], next_round_order))

            assert len(self._seats) == self._player_count

        players_for_round = [self._players[i] for i in self._seats]
        self._listener.end_round(players_for_round)
