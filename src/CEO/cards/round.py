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
        self._ceo_to_bottom = False
        self._ceo_final_cards = None

        assert len(self._players) == len(self._hands)

    def play(self):
        """
        Play a round.
        All behavoirs must be specified
        """
        gen = self.play_generator()
        gen_results = list(gen)
        assert gen_results == []

    def play_generator(self):
        """
        Generator version of play that allows asynchronous behaviors.
        """
        starting_player = 0
        trick_number = 0
        while not self._all_cards_played():
            trick_number += 1
            assert trick_number < len(self._hands) * 13

            starting_player = yield from self._play_trick(starting_player)

        next_round_players = [self._players[i] for i in self.get_next_round_order()]
        self._listener.end_round(next_round_players)

    def _play_trick(self, starting_player: int):
        # Calculate the order of the other players after the player that leads.
        # Note that play_order is an iterator.
        play_order = map(
            (lambda x: (x + starting_player) % self._player_count),
            range(self._player_count),
        )
        next(play_order)

        # Handle the lead
        cur_player = self._players[starting_player]
        cur_hand = self._hands[starting_player]

        # Validate and diagnostic logging.
        if cur_hand.is_empty():
            print(f"Error: No cards in leader's hand. Starting player {starting_player}")
            print(f"cur_player.behavior.is_rl {cur_player.behavoir.is_reinforcement_learning}")
            for hand in self._hands:
                print(hand)
        assert not cur_hand.is_empty()

        # Run the round.
        state = RoundState()
        state.initialize(self._hands, None)

        self._listener.before_lead(starting_player, cur_player, cur_hand, state)

        if not cur_player.behavoir.is_reinforcement_learning:
            cur_card_value = cur_player.behavoir.lead(starting_player, cur_hand, state)
            assert cur_card_value is not None
        else:
            cur_card_value = yield "lead", starting_player, cur_hand, state
            assert cur_card_value is not None

        assert cur_card_value is not None

        cur_card_count = cur_hand.count(cur_card_value)

        self._listener.lead(cur_card_value, cur_card_count, starting_player, cur_player)

        self._play_cards(starting_player, cur_card_value, cur_card_count)

        if self._hands[starting_player].is_empty():
            self._check_ceo_done()

        # Debugging
        if cur_card_count <= 0:
            print("Error: Player", cur_player.name, "lead no cards")

        assert cur_card_value is not None
        assert cur_card_count > 0

        # Let the remaining players play
        last_index_to_play = starting_player
        for cur_index in play_order:
            cur_player = self._players[cur_index]
            cur_hand = self._hands[cur_index]

            if cur_hand.is_empty():
                continue

            state = RoundState()
            state.initialize(self._hands, last_index_to_play)

            self._listener.before_play_cards(
                starting_player,
                cur_index,
                cur_player,
                cur_hand,
                cur_card_value,
                cur_card_count,
                state,
            )

            if not cur_player.behavoir.is_reinforcement_learning:
                new_card_value = cur_player.behavoir.play_on_trick(
                    starting_player,
                    cur_index,
                    cur_hand,
                    cur_card_value,
                    cur_card_count,
                    state,
                )
            else:
                new_card_value = (
                    yield "play",
                    starting_player,
                    cur_index,
                    cur_hand,
                    cur_card_value,
                    cur_card_count,
                    state,
                )

            if new_card_value is None:
                self._listener.pass_on_trick(cur_index, cur_player)
                continue

            # Debugging
            if new_card_value.value <= cur_card_value.value:
                print(
                    "Player",
                    cur_player.name,
                    "plays",
                    new_card_value,
                    "on",
                    cur_card_value,
                )
            assert new_card_value.value > cur_card_value.value

            self._play_cards(cur_index, new_card_value, cur_card_count)

            if self._hands[cur_index].is_empty():
                self._check_ceo_done()

            # print(cur_index, " ", cur_player, " plays ", new_card_value)
            self._listener.play_cards(new_card_value, cur_card_count, cur_index, cur_player)

            cur_card_value = new_card_value
            last_index_to_play = cur_index

        # If the last player to play on the trick went out, then
        # find the next player to lead.
        orig_last_index_to_play = last_index_to_play
        if self._hands[last_index_to_play].is_empty():
            last_index_to_play = 0
            while (
                last_index_to_play < self._player_count
                and self._hands[last_index_to_play].is_empty()
            ):
                last_index_to_play += 1

        if last_index_to_play == len(self._hands):
            for hand in self._hands:
                assert hand.is_empty()
        else:
            if self._hands[last_index_to_play].is_empty():
                print(
                    f"Error: next to play is empty {last_index_to_play} orig {orig_last_index_to_play} player count {self._player_count}"
                )
                for hand in self._hands:
                    print(hand)
            assert not self._hands[last_index_to_play].is_empty()
        return last_index_to_play

    def _check_ceo_done(self):
        """
        Another player went out. See if CEO is still playing
        """
        if self._hands[0].is_empty():
            return

        # CEO isn't out, so they lost and don't have to play anymore
        self._ceo_to_bottom = True
        self._ceo_final_cards = 0

        ceo_hand = self._hands[0]
        for i in range(13):
            cv = CardValue(i)
            count = ceo_hand.count(cv)

            if count > 0:
                self._hands[0].remove_cards(cv, count)
                self._ceo_final_cards += count

    def get_next_round_order(self) -> list[int]:
        if len(self._next_round_order) == len(self._players):

            if self._next_round_order[0] == 0:
                self._ceo_final_cards = 0

            return self._next_round_order

        # Construct the next round order
        if self._ceo_to_bottom:
            self._next_round_order.append(0)
            self._ceo_to_bottom = False
        else:
            self._ceo_final_cards = 0

        return self._next_round_order

    def get_final_ceo_card_count(self) -> int:
        """Returns the number of cards in CEO's hand at the end of the round.
        This will be positive if CEO loses.
        """
        return self._ceo_final_cards

    def _play_cards(self, player_index: int, card_value: CardValue, count: int):
        theHand = self._hands[player_index]
        theHand.remove_cards(card_value, count)

        if theHand.is_empty():
            self._next_round_order.append(player_index)

    def _all_cards_played(self):
        return all(map((lambda x: x.is_empty()), self._hands))
