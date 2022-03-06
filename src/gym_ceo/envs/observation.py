import numpy as np

from CEO.cards.hand import Hand, CardValue


class ObservationFactory:
    """Class that creates Observations"""

    # The indices into the observation array where the various features start
    _obs_index_hand_cards: int
    _obs_index_other_player_card_count: int
    _obs_index_other_player_card_count: int
    _obs_index_cur_trick_value: int
    _obs_index_cur_trick_count: int
    _obs_index_start_player: int

    _num_players: int

    observation_dimension: int

    def __init__(self, num_players: int):
        # Thirteen dimensions for the cards in the hand.
        # (num_players - 1) dimensions for the number of cards in the other player's hands
        # One dimension for the current value of the trick
        # One dimension for the number of cards in the trick
        # One dimension for the starting player on the trick
        self._obs_index_hand_cards = 0
        self._obs_index_other_player_card_count = self._obs_index_hand_cards + 13
        self._obs_index_cur_trick_value = self._obs_index_other_player_card_count + num_players - 1
        self._obs_index_cur_trick_count = self._obs_index_cur_trick_value + 1
        self._obs_index_start_player = self._obs_index_cur_trick_count + 1

        self._num_players = num_players

        self.observation_dimension = self._obs_index_start_player + 1

    def create_observation(self, **kwargs):
        """Creates an observation. See Observation constructor for a description of
        the arguments.
        """
        return Observation(self, **kwargs)


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

        To create an observation from another observation while updating the cards in the
        hand:
        array: ndarray for the original observation
        update_hand: HandInterface object giving the hand

        To create an observation from an ndarray:
        array: ndarray for the observation.
        """

        self._factory = factory

        obs_index_hand_cards = self._factory._obs_index_hand_cards
        obs_index_other_player_card_count = self._factory._obs_index_other_player_card_count
        obs_index_cur_trick_value = self._factory._obs_index_cur_trick_value
        obs_index_cur_trick_count = self._factory._obs_index_cur_trick_count
        obs_index_start_player = self._factory._obs_index_start_player
        num_players = self._factory._num_players

        if "type" in kwargs and kwargs["type"] == "lead":
            cur_hand = kwargs["cur_hand"]
            starting_player = kwargs["starting_player"]
            cur_hand = kwargs["cur_hand"]
            state = kwargs["state"]

            # Create the return array
            self._obs = np.zeros(self._factory.observation_dimension)

            # Add the cards in our hand
            for v in range(13):
                self._obs[obs_index_hand_cards + v] = cur_hand.count(CardValue(v))

            # Add the cards in other players' hands. Don't include the agent's hand.
            for p in range(1, num_players):
                self._obs[obs_index_other_player_card_count + p - 1] = state.cards_remaining[p]

            # Add the trick state
            self._obs[obs_index_cur_trick_value] = 0
            self._obs[obs_index_cur_trick_count] = 0
            self._obs[obs_index_start_player] = 0

        elif "type" in kwargs and kwargs["type"] == "play":
            cur_hand = kwargs["cur_hand"]
            starting_player = kwargs["starting_player"]
            cur_hand = kwargs["cur_hand"]
            cur_card_value = kwargs["cur_card_value"]
            cur_card_count = kwargs["cur_card_count"]
            state = kwargs["state"]

            # Create the return array
            self._obs = np.zeros(self._factory.observation_dimension)

            # Add the cards in our hand
            for v in range(13):
                self._obs[obs_index_hand_cards + v] = cur_hand.count(CardValue(v))

            # Add the cards in other players' hands. Don't include the agent's hand.
            for p in range(1, num_players):
                self._obs[obs_index_other_player_card_count + p - 1] = state.cards_remaining[p]

            # Add the trick state
            self._obs[obs_index_cur_trick_value] = cur_card_value.value
            self._obs[obs_index_cur_trick_count] = cur_card_count
            self._obs[obs_index_start_player] = starting_player

        elif "update_hand" in kwargs:
            update_hand = kwargs["update_hand"]

            # Create the return array
            self._obs = kwargs["array"].copy()

            # Update the cards in the hand
            for v in range(13):
                self._obs[obs_index_hand_cards + v] = update_hand.count(CardValue(v))

        elif "array" in kwargs:
            assert isinstance(kwargs["array"], np.ndarray)
            self._obs = kwargs["array"]

        else:
            raise Exception("Illegal arguments")

    def get_card_count(self, card_value: int):
        return self._obs[self._factory._obs_index_hand_cards + card_value]

    def get_other_player_card_count(self, player_index: int):
        return self._obs[self._factory._obs_index_other_player_card_count + player_index]

    def get_starting_player(self):
        return self._obs[self._factory._obs_index_start_player]

    def get_cur_trick_count(self):
        return self._obs[self._factory._obs_index_cur_trick_count]

    def get_cur_trick_value(self):
        # See if we lead
        if self.get_cur_trick_count() == 0:
            return None

        return self._obs[self._factory._obs_index_cur_trick_value]

    def get_array(self):
        return self._obs

    def copy(self):
        return Observation(self._factory, array=self._obs.copy())
