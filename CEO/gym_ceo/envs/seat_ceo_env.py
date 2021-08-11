import gym
import numpy as np
from gym import error, spaces, utils
from gym.spaces import Box, Discrete
from gym.utils import seeding

from gym_ceo.envs.actions import Actions

from CEO.cards.round import Round, RoundState
from CEO.cards.eventlistener import EventListenerInterface
from CEO.cards.deck import Deck
from CEO.cards.hand import Hand, CardValue
from CEO.cards.simplebehavior import BasicBehavior
from CEO.cards.player import Player


class SeatCEOEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    observation_space = Box(
        low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        high=np.array([13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13]),
        dtype=np.int32,
    )

    _num_players: int
    _round: Round
    _listener: EventListenerInterface
    _hands: list[Hand]
    _observation_dimension: int
    _actions: Actions
    _info = dict()

    _action_space_lead = Discrete(Actions.action_lead_count)
    _action_space_play = Discrete(Actions.action_play_count)

    _cur_hand: Hand
    _cur_trick_count: int
    _cur_trick_value: CardValue

    def __init__(self, num_players=6, behaviors=[], hands=[], listener=EventListenerInterface()):
        self._num_players = num_players

        if listener is None:
            self._listener = EventListenerInterface()
        else:
            self._listener = listener

        self._actions = Actions()

        self._hands = hands

        self._players = []
        self._players.append(Player("RL", None))
        for i in range(num_players - 1):
            name = "Basic" + str(i + 1)

            if i >= len(behaviors):
                self._players.append(Player(name, BasicBehavior()))
            else:
                self._players.append(Player(name, behaviors[i]))

        # Thirteen dimensions for the cards in the hand.
        # (num_players - 1) dimensions for the number of cards in the other player's hands
        # One dimension for the current value of the trick
        # One dimension for the number of cards in the trick
        # One dimension for the starting player on the trick
        self._observation_dimension = 13 + num_players - 1 + 3
        self.observation_space = Box(
            low=np.array([0] * self._observation_dimension),
            high=np.array([13] * self._observation_dimension),
            dtype=np.int32,
        )

        self.action_space = self._action_space_play

    def reset(self):
        self._listener.start_round(self._players)
        self.action_space = self._action_space_lead

        if self._hands is None or len(self._hands) == 0:
            deck = Deck(self._num_players)
            self._hands = deck.deal()

        self._round = Round(self._players, self._hands, self._listener)
        self._gen = self._round.play_generator()

        gen_tuple = next(self._gen)
        assert gen_tuple[0] == "lead"

        return self._make_observation(gen_tuple)

    def step(self, action):
        assert (
            isinstance(action, int) or isinstance(action, np.int32) or isinstance(action, np.int64)
        )

        cv = self._actions.play(
            self._cur_hand, self._cur_trick_value, self._cur_trick_count, action
        )

        if cv is None and self._cur_trick_value is None:
            print("Action", action, "returned None to play on trick. Hand", hand)
            assert cv is not None

        reward = 0
        done = False
        try:
            while True:
                gen_tuple = self._gen.send(cv)

                obs = self._make_observation(gen_tuple)

                if obs is not None:
                    break
                else:
                    # The only action is to pass
                    cv = None

            done = False
        except StopIteration:
            done = True
            obs = None
            next_round_order = self._round.get_next_round_order()

            if next_round_order[0] == 0:
                reward = 1.0
            else:
                reward = -1.0

            self._hands = None

        return obs, reward, done, self._info

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def _make_observation(self, gen_tuple):
        if gen_tuple[0] == "lead":
            self.action_space = self._action_space_lead
            return self._make_observation_lead(gen_tuple)
        elif gen_tuple[0] == "play":
            self.action_space = self._action_space_play
            return self._make_observation_play(gen_tuple)
        else:
            assert "Unexpected action" == ""

    def _make_observation_lead(self, gen_tuple):
        type_str, starting_player, cur_hand, state = gen_tuple

        self._cur_trick_count = None
        self._cur_trick_value = None
        self._cur_hand = cur_hand

        # Create the return array
        obs = np.zeros(self._observation_dimension)
        i = 0

        # Add the cards in our hand
        for v in range(13):
            obs[i] = cur_hand.count(CardValue(v))
            i += 1

        # Add the cards in other players' hands
        for p in range(1, self._num_players):
            obs[i] = state.cards_remaining[p]
            i += 1

        # Add the trick state
        obs[i] = 0
        i += 1

        obs[i] = 0
        i += 1

        obs[i] = 0
        i += 1

        assert i == self._observation_dimension

        return obs

    def _make_observation_play(self, gen_tuple):
        (
            type_str,
            starting_player,
            cur_index,
            cur_hand,
            cur_card_value,
            cur_card_count,
            state,
        ) = gen_tuple

        self._cur_trick_count = cur_card_count
        self._cur_trick_value = cur_card_value
        self._cur_hand = cur_hand

        # See if we must pass, i.e., there is no choice of action
        if cur_hand.max_card_value().value <= cur_card_value.value:
            return None

        # Create the return array
        obs = np.zeros(self._observation_dimension)
        i = 0

        # Add the cards in our hand
        for v in range(13):
            obs[i] = cur_hand.count(CardValue(v))
            i += 1

        # Add the cards in other players' hands
        for p in range(1, self._num_players):
            obs[i] = state.cards_remaining[p]
            i += 1

        # Add the trick state
        obs[i] = cur_card_value.value
        i += 1

        obs[i] = cur_card_count
        i += 1

        obs[i] = starting_player
        i += 1

        assert i == self._observation_dimension

        return obs
