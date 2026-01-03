import numpy as np
import torch as th

from CEO.cards.hand import CardValue, HandInterface, PlayedCards


class ObservationFactory:
    """Class that creates Observations"""

    # The indices into the observation array where the various features start
    _obs_index_hand_cards: int
    _obs_index_other_player_card_count: int
    _obs_index_cur_trick_value: int
    _obs_index_cur_trick_count: int
    _obs_index_start_player: int
    _obs_index_last_player: int

    _obs_index_pass_valid: int
    _obs_index_play_value_0_valid: int

    _obs_value_trick_not_started: int

    _num_players: int
    _seat_number: int
    """The seat number of the agent. Zero is CEO."""

    observation_dimension: int

    def __init__(self, num_players: int, seat_number: int, include_valid_actions=False):
        # Thirteen dimensions for the cards in the hand.
        # (num_players - 1) dimensions for the number of cards
        # in the other player's hands
        # One dimension for the current value of the trick
        # One dimension for the number of cards in the trick
        # One dimension for the starting player on the trick
        # One dimension for the last player that played on the trick.
        self._obs_index_hand_cards = 0
        self._obs_index_other_player_card_count = self._obs_index_hand_cards + 13
        self._obs_index_cur_trick_value = (
            self._obs_index_other_player_card_count + num_players - 1
        )
        self._obs_index_cur_trick_count = self._obs_index_cur_trick_value + 1
        self._obs_index_start_player = self._obs_index_cur_trick_count + 1
        self._obs_index_last_player = self._obs_index_start_player + 1

        if include_valid_actions:
            self._obs_index_play_value_0_valid = self._obs_index_last_player + 1
            self._obs_index_pass_valid = self._obs_index_play_value_0_valid + 13

            self.observation_dimension = self._obs_index_pass_valid + 1
        else:
            self._obs_index_pass_valid = None
            self._obs_index_play_value_0_valid = None

            self.observation_dimension = self._obs_index_last_player + 1

        self._obs_value_trick_not_started = num_players

        self._num_players = num_players
        self._seat_number = seat_number

    def create_observation(self, **kwargs):
        """Creates an observation. See Observation constructor for a description of
        the arguments.
        """
        return Observation(self, **kwargs)

    def get_valid_action_range(self):
        return (
            self._obs_index_play_value_0_valid,
            self._obs_index_play_value_0_valid + 14,
        )


class Observation:
    """Class implementing a full observation of the CEO state.
    The observation includes all information available to a player.
    """

    _factory: ObservationFactory
    _obs: np.ndarray

    def __init__(self, factory: ObservationFactory, **kwargs):
        """Create an observation
        To create an observation from a state where the player should lead:
        type: "lead"
        starting_player: integer player index
        cur_hand: Hand object giving the players hand
        state: RoundState object

        To create an observation from a state where the player should play on a trick:
        type: "play"
        starting_player: Integer giving the seat of the starting player.
        cur_index: Integer giving the seat of the player
        cur_hand: Hand object giving the players hand
        cur_card_value: CardValue object giving the value of the last played
                        card in the trick.
        cur_card_count: The number of cards that must be played on the trick.
        state: RoundState object

        To create an observation from another observation while updating
        the cards in the hand:
        array: ndarray for the original observation
        update_hand: HandInterface object giving the hand

        To create an observation from an ndarray:
        array: ndarray for the observation.

        To create an observation from a pytorch Tensor:
        tensor: th.Tensor for the observation.
        """

        self._factory = factory

        obs_index_hand_cards = self._factory._obs_index_hand_cards
        obs_index_other_player_card_count = (
            self._factory._obs_index_other_player_card_count
        )
        obs_index_cur_trick_value = self._factory._obs_index_cur_trick_value
        obs_index_cur_trick_count = self._factory._obs_index_cur_trick_count
        obs_index_start_player = self._factory._obs_index_start_player
        obs_index_last_player = self._factory._obs_index_last_player

        obs_value_trick_not_started = self._factory._obs_value_trick_not_started

        num_players = self._factory._num_players
        seat_number = self._factory._seat_number

        if "type" in kwargs and kwargs["type"] == "lead":
            cur_hand = kwargs["cur_hand"]
            starting_player = kwargs["starting_player"]
            state = kwargs["state"]

            # Create the return array
            self._obs = np.zeros(self._factory.observation_dimension)

            # Add the cards in our hand
            for v in range(13):
                self._obs[obs_index_hand_cards + v] = cur_hand.count(CardValue(v))

            # Add the cards in other players' hands. Don't include the agent's hand.
            seat_iter = [p1 for p1 in range(num_players) if p1 != seat_number]
            for p, i in zip(seat_iter, range(0, num_players - 1), strict=False):
                self._obs[obs_index_other_player_card_count + i] = (
                    state.cards_remaining[p]
                )

            # Add the trick state
            self._obs[obs_index_cur_trick_value] = 0
            self._obs[obs_index_cur_trick_count] = 0
            self._obs[obs_index_start_player] = 0
            self._obs[obs_index_last_player] = obs_value_trick_not_started

            # Add in valid actions, if necessary
            if self._factory._obs_index_pass_valid is not None:
                self._add_valid_actions_to_observation_lead(cur_hand)

        elif "type" in kwargs and kwargs["type"] == "play":
            cur_hand = kwargs["cur_hand"]
            starting_player = kwargs["starting_player"]
            cur_card_value = kwargs["cur_card_value"]
            cur_card_count = kwargs["cur_card_count"]
            state = kwargs["state"]

            # Create the return array
            self._obs = np.zeros(self._factory.observation_dimension)

            # Add the cards in our hand
            for v in range(13):
                self._obs[obs_index_hand_cards + v] = cur_hand.count(CardValue(v))

            # Add the cards in other players' hands. Don't include the agent's hand.
            seat_iter = [p1 for p1 in range(num_players) if p1 != seat_number]
            for p, i in zip(seat_iter, range(0, num_players - 1), strict=False):
                self._obs[obs_index_other_player_card_count + i] = (
                    state.cards_remaining[p]
                )

            # Add the trick state
            self._obs[obs_index_cur_trick_value] = cur_card_value.value
            self._obs[obs_index_cur_trick_count] = cur_card_count
            self._obs[obs_index_start_player] = starting_player
            self._obs[obs_index_last_player] = state.last_player_to_play_index

            # Add in valid actions, if necessary
            if self._factory._obs_index_pass_valid is not None:
                self._add_valid_actions_to_observation_play(
                    cur_hand, cur_card_count, cur_card_value
                )

        elif "update_hand" in kwargs and "update_played_cards" in kwargs:
            update_hand = kwargs["update_hand"]
            played_cards: PlayedCards = kwargs["update_played_cards"]
            update_last_player: int = kwargs["update_last_player"]

            # Create the return array
            self._obs = kwargs["array"].copy()

            # Update the cards in the hand
            for v in range(13):
                self._obs[obs_index_hand_cards + v] = update_hand.count(CardValue(v))

            # Update the played cards
            self._obs[obs_index_cur_trick_value] = played_cards.value.value
            self._obs[obs_index_cur_trick_count] = played_cards.count
            self._obs[obs_index_last_player] = update_last_player

        elif "array" in kwargs:
            assert isinstance(kwargs["array"], np.ndarray), (
                f"Expected ndarray but got {type(kwargs['array'])}"
            )
            self._obs = kwargs["array"]

        elif "tensor" in kwargs:
            assert isinstance(kwargs["tensor"], th.Tensor)
            self._obs = kwargs["tensor"]

        else:
            raise Exception("Illegal arguments: " + str(kwargs))

    def _add_valid_actions_to_observation_lead(self, hand: HandInterface):
        self._obs[self._factory._obs_index_pass_valid] = 0.0

        for card_value in range(13):
            count = self._obs[self._factory._obs_index_hand_cards + card_value]
            self._obs[self._factory._obs_index_play_value_0_valid + card_value] = (
                1.0 if count > 0 else 0.0
            )

    def _add_valid_actions_to_observation_play(
        self, hand: HandInterface, cur_trick_count: int, cur_trick_value: CardValue
    ):
        self._obs[self._factory._obs_index_pass_valid] = 1.0

        for card_value in range(13):
            if card_value <= cur_trick_value.value:
                self._obs[self._factory._obs_index_play_value_0_valid + card_value] = (
                    0.0
                )
                continue

            count = self._obs[self._factory._obs_index_hand_cards + card_value]

            self._obs[self._factory._obs_index_play_value_0_valid + card_value] = (
                1.0 if count >= cur_trick_count else 0.0
            )

    def get_card_count(self, card_value: int):
        return self._obs[self._factory._obs_index_hand_cards + card_value]

    def get_other_player_card_count(self, adj_player_index: int):
        """Returns the number of cards in another player's hand. The index is
        the position of the other player in a list containing all players except for
        the agent."""
        return self._obs[
            self._factory._obs_index_other_player_card_count + adj_player_index
        ]

    def get_starting_player(self):
        return self._obs[self._factory._obs_index_start_player]

    def get_cur_trick_count(self):
        return self._obs[self._factory._obs_index_cur_trick_count]

    def get_cur_trick_value(self):
        # See if we lead
        if self.get_cur_trick_count() == 0:
            return None

        return self._obs[self._factory._obs_index_cur_trick_value]

    def get_last_player(self):
        """Returns the index of the last player to play on the trick. None is returned
        if the trick hasn't started."""

        if False:
            print(self._obs)
            print("len", len(self._obs))
            print("index", self._factory._obs_index_last_player)
        value = self._obs[self._factory._obs_index_last_player]

        if value == self._factory._obs_value_trick_not_started:
            return None

        return value

    def get_pass_action_valid(self):
        if self._factory._obs_index_pass_valid is None:
            raise Exception("Can't call get_pass_action_valid")

        return self._obs[self._factory._obs_index_pass_valid]

    def get_play_card_action_valid(self, card_value: int):
        if self._factory._obs_index_pass_valid is None:
            raise Exception("Can't call get_pass_action_valid")

        return self._obs[self._factory._obs_index_play_value_0_valid + card_value]

    def has_playable_card_action(self):
        if self._factory._obs_index_play_value_0_valid is None:
            # We aren't configured to include valid actions in the observation.
            return True

        for i in range(13):
            if self._obs[self._factory._obs_index_play_value_0_valid + i] > 0.0:
                return True

        return False

    def get_array(self):
        return self._obs

    def get_valid_action_array(self):
        start = self._factory._obs_index_play_value_0_valid
        end = start + 14
        return self._obs[start:end]

    def copy(self):
        return Observation(self._factory, array=self._obs.copy())
