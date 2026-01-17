from argparse import ArgumentError

import gymnasium
import numpy as np

from ceo.envs.actions import (
    ActionEnum,
    ActionSpaceFactory,
    AllCardActionSpaceFactory,
    CardActionSpaceFactory,
    CEOActionSpace,
)
from ceo.envs.observation import Observation, ObservationFactory
from ceo.envs.observation_hand import ObservationHand
from ceo.envs.rl_behavior import RLBehavior
from ceo.game.deck import Deck
from ceo.game.eventlistener import EventListenerInterface
from ceo.game.hand import CardValue, Hand, PlayedCards
from ceo.game.passcards import PassCards
from ceo.game.player import Player
from ceo.game.round import Round
from ceo.game.simplebehavior import BasicBehavior, SimpleBehaviorBase


class CEOAnySeatEnv(gymnasium.Env):
    """
    Environment where the agent's seat changes randomly each episode.
    The seat number is included in the observation so the agent knows
    which position it is playing from.
    """

    metadata = {"render.modes": ["human"]}

    action_space: CEOActionSpace
    _action_space_factory: ActionSpaceFactory

    # The maximum possible action space size
    max_action_space_size: int

    action_space_type: str

    num_players: int

    _current_seat: int
    """The seat for the current episode. Changes each reset."""

    _round: Round
    _listener: EventListenerInterface
    _hands: list[Hand]
    _observation_dimension: int
    _info = dict()

    _next_reset_hands: list[Hand]
    """Hands to be used the next time reset is called."""

    _skip_passing: bool

    _cur_hand: Hand
    _cur_trick_count: int
    _cur_trick_value: CardValue

    _simple_behavior_base: SimpleBehaviorBase

    _reward_includes_cards_left: bool

    observation_factory: ObservationFactory

    _behaviors: list
    """Behaviors for non-agent players. None at the agent's seat."""

    def __init__(
        self,
        num_players=6,
        behaviors=None,
        hands=None,
        listener=None,
        skip_passing=False,
        *,
        action_space_type="ceo",
        reward_includes_cards_left=False,
        obs_kwargs=None,
    ):
        self.num_players = num_players
        self._skip_passing = skip_passing
        self._reward_includes_cards_left = reward_includes_cards_left

        # Initialize defaults inside the function to avoid mutable/default-call issues
        if behaviors is None:
            behaviors = [BasicBehavior() for _ in range(num_players)]

        assert len(behaviors) == num_players
        self._behaviors = behaviors

        if listener is None:
            self._listener = EventListenerInterface()
        else:
            self._listener = listener

        self._next_reset_hands = hands
        assert (
            self._next_reset_hands is None
            or len(self._next_reset_hands) == self.num_players
        )

        if obs_kwargs is None:
            obs_kwargs = dict()

        # Create observation factory with seat number included
        self.observation_factory = ObservationFactory(
            num_players, include_seat_number=True, **obs_kwargs
        )
        self._observation_dimension = self.observation_factory.observation_dimension
        self.observation_space = self.observation_factory.create_observation_space()

        self.action_space_type = action_space_type
        if action_space_type == "ceo":
            # Use the action space with a limited number of choices
            self._action_space_factory = ActionSpaceFactory()
            self.max_action_value = len(ActionEnum)
        elif action_space_type == "card":
            # Use the action space where all cards in the hand can be played
            self._action_space_factory = CardActionSpaceFactory()
            self.max_action_value = 13
        elif action_space_type == "all_card":
            # Use the action space with constant size where all cards can be played and
            # invalid actions are clipped or cause the episode to end.
            self._action_space_factory = AllCardActionSpaceFactory()
            self.max_action_value = 14
        else:
            raise ArgumentError("Invalid action_space_type: ", action_space_type)

        self.action_space = self._action_space_factory.default_lead()

        self._simple_behavior_base = SimpleBehaviorBase()

        # Initialize current seat (will be set properly in reset)
        self._current_seat = 0

    def _create_players_for_seat(self, seat: int) -> list[Player]:
        """Creates the player list with RLBehavior at the given seat."""
        players = []
        for i in range(self.num_players):
            if i == seat:
                players.append(Player("RL", RLBehavior()))
            else:
                name = f"Player{i}"
                players.append(Player(name, self._behaviors[i]))
        return players

    def _update_observation_factory_seat(self, seat: int):
        """Updates the observation factory to use the given seat."""
        self.observation_factory.set_seat_number(seat)

    def _select_seat(self) -> int:
        """Selects the seat for the episode. Override in subclasses for fixed seat."""
        return int(self.np_random.integers(0, self.num_players))

    def reset(self, *, seed=None, options=None, hands: list[Hand] = None):
        super().reset(seed=seed)

        # Select seat for this episode (random in base class, fixed in subclasses)
        self._current_seat = self._select_seat()

        # Update observation factory with new seat
        self._update_observation_factory_seat(self._current_seat)

        # Create players with RLBehavior at the selected seat
        self._players = self._create_players_for_seat(self._current_seat)

        self._listener.start_round(self._players)

        # Deal the cards
        if hands is not None:
            self._hands = hands
        elif self._next_reset_hands is not None:
            self._hands = self._next_reset_hands
            self._next_reset_hands = None
            assert len(self._hands) == len(self._players)
        else:
            deck = Deck(self.num_players)
            self._hands = deck.deal()

        # Pass cards
        if not self._skip_passing:
            orig_card_counts = list([hand.card_count() for hand in self._hands])

            passcards = PassCards(self._players, self._hands, self._listener)
            passcards.do_card_passing()

            # Validate
            for hand, orig_count in zip(self._hands, orig_card_counts, strict=False):
                assert hand.card_count() == orig_count

        # Start the round.
        self._round = Round(self._players, self._hands, self._listener)
        self._gen = self._round.play_generator()

        self.action_space = self._action_space_factory.create_lead(
            self._hands[self._current_seat]
        )

        gen_tuple = next(self._gen)

        obs, reward, done, truncated, info = self._play_until_action_needed(gen_tuple)

        assert not done
        assert reward == 0.0

        return obs, info

    def step(self, action):
        assert isinstance(action, (int, np.int32, np.int64))

        if action >= self.action_space.n:
            print(f"Error: action {action} is larger than {self.action_space}")
        assert action < self.action_space.n

        ret = self.action_space.card_to_play(
            self._cur_hand, self._cur_trick_value, self._cur_trick_count, action
        )

        action_reward = 0.0
        if ret is None:
            cv = None
        elif isinstance(ret, CardValue):
            cv = ret
        elif isinstance(ret, tuple):
            cv = ret[0]
            action_reward = ret[1]
        else:
            assert ("Unknown action type: " + type(ret)) == ""

        # If the reward is negative, then the action isn't valid. End the episode.
        if action_reward != 0:
            self._hands = None

            # Gym requires an np array for the observation here.
            obs = np.zeros(self._observation_dimension)

            return obs, action_reward, True, False, self._info

        if cv is None and self._cur_trick_value is None:
            print(
                "Action",
                action,
                "returned None to play on trick. Hand",
                self._cur_hand,
            )
            assert cv is not None

        try:
            # Perform the action
            gen_tuple = self._gen.send(cv)
        except StopIteration:
            return self._end_round()

        return self._play_until_action_needed(gen_tuple)

    def _play_until_action_needed(self, gen_tuple: tuple):
        """Advances the round until an action from the agent is needed. In particular,
        if the only action is to pass, then just pass instead of asking the agent for
        the action. The parameter is the tuple returned from performing the action.
        The method will return immediately if the tuple implies that an action is needed
        now."""

        reward = 0.0
        done = False
        try:
            while True:
                obs = self._make_observation(gen_tuple)

                if obs is not None:
                    obs_obj = self.observation_factory.create_observation(array=obs)
                    if obs_obj.has_playable_card_action():
                        break

                else:
                    # The only action is to pass
                    pass

                # Pass on the trick.
                gen_tuple = self._gen.send(None)

            done = False
        except StopIteration:
            return self._end_round()

        return obs, reward, done, False, self._info

    def _end_round(self):
        done = True
        obs = None
        next_round_order = self._round.get_next_round_order()

        next_round_position = next_round_order.index(self._current_seat)

        if self._current_seat == 0 and next_round_position == 0:
            # CEO stays in CEO
            reward = 1.0
        elif (
            self._current_seat == self.num_players - 1
            and next_round_position == self.num_players - 1
        ):
            # Bottom stays in bottom
            reward = 0.0
        elif self._current_seat == next_round_position:
            # Same position
            reward = 0.0
        elif self._current_seat > next_round_position:
            # Move up
            reward = 1.0
        else:
            # Move down
            assert self._current_seat < next_round_position
            reward = -1.0

            if self._reward_includes_cards_left:
                assert self._current_seat == 0
                reward -= self._round.get_final_ceo_card_count() / 13.0

        self._hands = None

        info = {"ceo_stay": next_round_order[0] == 0, "seat": self._current_seat}

        return obs, reward, done, False, info

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def _make_observation(self, gen_tuple):
        if gen_tuple[0] == "lead":
            obs = self._make_observation_lead(gen_tuple)
        elif gen_tuple[0] == "play":
            obs = self._make_observation_play(gen_tuple)
        else:
            raise Exception("Unexpected action")

        if obs is None:
            return None

        return obs.get_array()

    def _make_observation_lead(self, gen_tuple):
        type_str, starting_player, cur_hand, state = gen_tuple

        self._cur_trick_count = None
        self._cur_trick_value = None
        self._cur_hand = cur_hand
        self._info["hand"] = cur_hand

        # Set up the action space
        self.action_space = self._action_space_factory.create_lead(cur_hand)

        return Observation(
            self.observation_factory,
            type="lead",
            starting_player=starting_player,
            cur_hand=cur_hand,
            state=state,
        )

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
        self._info["hand"] = cur_hand

        # Set up the action space
        self.action_space = self._action_space_factory.create_play(
            cur_hand, cur_card_value, cur_card_count
        )
        if self.action_space is None:
            return None

        return Observation(
            self.observation_factory,
            type="play",
            starting_player=starting_player,
            cur_index=cur_index,
            cur_card_value=cur_card_value,
            cur_card_count=cur_card_count,
            cur_hand=cur_hand,
            state=state,
        )

    def get_afterstate(
        self, observation_array: np.ndarray, action
    ) -> tuple[np.ndarray, CardValue]:
        """Creates the afterstate from taking the given action from the state
        given by the observation."""

        # Create an observation and a hand from the array
        observation = self.observation_factory.create_observation(
            array=observation_array
        )
        hand = ObservationHand(observation)

        cur_trick_count = observation.get_cur_trick_count()
        if observation.get_cur_trick_value() is not None:
            cur_trick_value = CardValue(int(observation.get_cur_trick_value()))
        else:
            cur_trick_value = None

        # Get the card to play
        played_value = self.action_space.card_to_play(
            hand, cur_trick_value, cur_trick_count, action
        )

        # See if we pass
        if played_value is None:
            return observation_array.copy(), None

        # Perform the action on the hand
        if cur_trick_count == 0:
            cur_trick_count = hand.count(played_value)

        played_cards = PlayedCards(played_value, cur_trick_count)

        hand.play_cards(played_cards)

        # Create an observation for the afterstate
        afterstate = self.observation_factory.create_observation(
            array=observation_array,
            update_hand=hand,
            update_played_cards=played_cards,
            update_last_player=0,
        )

        return afterstate.get_array(), played_value
