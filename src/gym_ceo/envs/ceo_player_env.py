from argparse import ArgumentError
import gym
import numpy as np
from gym.spaces import Box

from gym_ceo.envs.actions import (
    ActionSpaceFactory,
    AllCardActionSpaceFactory,
    CEOActionSpace,
    CardActionSpaceFactory,
    ActionEnum,
)
from gym_ceo.envs.observation import Observation, ObservationFactory
from gym_ceo.envs.observation_hand import ObservationHand
from gym_ceo.envs.rl_behavior import RLBehavior

from CEO.cards.round import Round, RoundState
from CEO.cards.eventlistener import EventListenerInterface
from CEO.cards.deck import Deck
from CEO.cards.hand import Hand, CardValue, PlayedCards
from CEO.cards.simplebehavior import BasicBehavior, SimpleBehaviorBase
from CEO.cards.player import Player, PlayerBehaviorInterface
from CEO.cards.passcards import PassCards


class CEOPlayerEnv(gym.Env):
    """
    Environment for a player in the CEO seat. This environment's observations
    contains all the information available to a player.
    """

    metadata = {"render.modes": ["human"]}

    action_space: CEOActionSpace
    _action_space_factory: ActionSpaceFactory

    # The maximum possible action space size
    max_action_space_size: int

    action_space_type: str

    num_players: int

    seat_number: int

    _round: Round
    _listener: EventListenerInterface
    _hands: list[Hand]
    _observation_dimension: int
    _info = dict()

    _skip_passing: bool

    _cur_hand: Hand
    _cur_trick_count: int
    _cur_trick_value: CardValue

    _simple_behavior_base: SimpleBehaviorBase

    _reward_includes_cards_left: bool

    observation_factory: ObservationFactory

    def __init__(
        self,
        num_players=6,
        behaviors=[],
        hands=[],
        listener=EventListenerInterface(),
        skip_passing=False,
        *,
        action_space_type="ceo",
        reward_includes_cards_left=False,
        obs_kwargs=None,
    ):
        self.num_players = num_players
        self.seat_number = 0
        self._skip_passing = skip_passing
        self._reward_includes_cards_left = reward_includes_cards_left

        if listener is None:
            self._listener = EventListenerInterface()
        else:
            self._listener = listener

        self._hands = hands

        self._players = []
        self._players.append(Player("RL", RLBehavior()))
        for i in range(num_players - 1):
            name = "Basic" + str(i + 1)

            if i >= len(behaviors):
                self._players.append(Player(name, BasicBehavior()))
            else:
                self._players.append(Player(name, behaviors[i]))

        if obs_kwargs is None:
            obs_kwargs = dict()

        self.observation_factory = ObservationFactory(num_players, **obs_kwargs)
        self._observation_dimension = self.observation_factory.observation_dimension

        self.observation_space = Box(
            low=np.array([0.0] * self._observation_dimension),
            high=np.array([13.0] * self._observation_dimension),
            dtype=np.float64,
        )

        self.action_space_type = action_space_type
        if action_space_type == "ceo":
            # Use the action space with a limited number of choices
            self._action_space_factory = ActionSpaceFactory()
            self.max_action_value = len(ActionEnum)
            print("Using CEO action space")
        elif action_space_type == "card":
            # Use the action space where all cards in the hand can be played
            self._action_space_factory = CardActionSpaceFactory()
            self.max_action_value = 13
            print("Using card action space")
        elif action_space_type == "all_card":
            # Use the action space with constant size where all cards can be played and
            # invalid actions are clipped or cause the episode to end.
            self._action_space_factory = AllCardActionSpaceFactory()
            self.max_action_value = 14
            print("Using all card action space")
        else:
            raise ArgumentError("Invalid action_space_type: ", action_space_type)

        self.action_space = self._action_space_factory.default_lead()

        self._simple_behavior_base = SimpleBehaviorBase()

    def reset(self, hands: list[Hand] = None):
        self._listener.start_round(self._players)

        # Deal the cards
        if hands is not None:
            self._hands = hands
        elif self._hands is None or len(self._hands) == 0:
            deck = Deck(self.num_players)
            self._hands = deck.deal()

        # Pass cards
        if not self._skip_passing:
            passcards = PassCards(self._players, self._hands, self._listener)
            passcards.do_card_passing()

        self._round = Round(self._players, self._hands, self._listener)
        self._gen = self._round.play_generator()

        self.action_space = self._action_space_factory.create_lead(self._hands[0])

        gen_tuple = next(self._gen)
        assert gen_tuple[0] == "lead"

        obs = self._make_observation(gen_tuple)

        if not self.observation_space.contains(obs):
            print("Obs", obs)
            print("len(obs)", len(obs))
            print("Obs space", self.observation_space)
            assert self.observation_space.contains(obs)

        return obs

    def step_full_action(self, full_action):
        self.step(self.action_space.find_full_action(full_action))

    def step(self, action):
        assert (
            isinstance(action, int) or isinstance(action, np.int32) or isinstance(action, np.int64)
        )

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

            # The gym environment checker wants to have an np array for the observation here.
            obs = np.zeros(self._observation_dimension)

            return obs, action_reward, True, self._info

        if cv is None and self._cur_trick_value is None:
            print(
                "Action",
                action,
                "returned None to play on trick. Hand",
                self._cur_hand,
            )
            assert cv is not None

        reward = 0.0
        done = False
        try:
            while True:
                gen_tuple = self._gen.send(cv)

                obs = self._make_observation(gen_tuple)

                # Check if pass is the only possible play.
                if obs is None:
                    # There aren't any playable cards
                    cv = None
                    continue

                obsObj = self.observation_factory.create_observation(array=obs)
                if not obsObj.has_playable_card_action():
                    # There aren't any playable cards
                    cv = None
                    continue

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
                reward += 1.0
            else:
                reward += -1.0

                if self._reward_includes_cards_left:
                    reward -= self._round.get_final_ceo_card_count() / 13.0

            self._hands = None

        return obs, reward, done, self._info

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
        if self.action_space == None:
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

    def get_afterstate(self, observation_array: np.ndarray, action) -> tuple[np.ndarray, CardValue]:
        """Creates the afterstate from taking the given action from the state
        given by the observation."""

        # Create an observation and a hand from the array
        observation = self.observation_factory.create_observation(array=observation_array)
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
