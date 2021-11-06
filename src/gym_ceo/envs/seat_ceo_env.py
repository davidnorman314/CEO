import gym
import numpy as np
from gym import error, spaces, utils
from gym.spaces import Box, Discrete
from gym.utils import seeding

from gym_ceo.envs.actions import Actions, ActionEnum
from gym_ceo.envs.observation import Observation, ObservationFactory

from CEO.cards.round import Round, RoundState
from CEO.cards.eventlistener import EventListenerInterface
from CEO.cards.deck import Deck
from CEO.cards.hand import Hand, CardValue
from CEO.cards.simplebehavior import BasicBehavior, SimpleBehaviorBase
from CEO.cards.player import Player, PlayerBehaviorInterface
from CEO.cards.passcards import PassCards


class RLBehavior(PlayerBehaviorInterface, SimpleBehaviorBase):
    """
    Class used for RL behavior
    """

    def __init__(self):
        self.is_reinforcement_learning = True

    def pass_cards(self, hand: Hand, count: int) -> list[CardValue]:
        return self.pass_singles(hand, count)

    def lead(self, player_position: int, hand: Hand, state: RoundState) -> CardValue:
        assert not "This should not be called"

    def play_on_trick(
        self,
        starting_position: int,
        player_position: int,
        hand: Hand,
        cur_trick_value: CardValue,
        cur_trick_count: int,
        state: RoundState,
    ) -> CardValue:
        assert not "This should not be called"


class CEOActionSpace(Discrete):
    actions: list[ActionEnum]

    def __init__(self, actions: list[int]):
        super(CEOActionSpace, self).__init__(len(actions))

        self.actions = actions

    def find_full_action(self, full_action: int) -> int:
        return self.actions.index(full_action)

    def __eq__(self, other):
        if not super(CEOActionSpace, self).__eq__(other):
            return False

        return self.actions == other.actions


class SeatCEOEnv(gym.Env):
    """
    Environment for a player in the CEO seat. This environment's observations
    contains all the information available to a player.
    """

    metadata = {"render.modes": ["human"]}

    observation_space = Box(
        low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        high=np.array([13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13]),
        dtype=np.int32,
    )

    action_space: CEOActionSpace

    # The maximum possible action space size
    max_action_space_size: int

    num_players: int

    _round: Round
    _listener: EventListenerInterface
    _hands: list[Hand]
    _observation_dimension: int
    _actions: Actions
    _info = dict()

    _skip_passing: bool

    action_space_lead = CEOActionSpace(
        [
            ActionEnum.PLAY_HIGHEST_NUM,
            ActionEnum.PLAY_SECOND_LOWEST_WITHOUT_BREAK_NUM,
            ActionEnum.PLAY_LOWEST_WITHOUT_BREAK_NUM,
        ]
    )
    action_space_one_legal_lead = CEOActionSpace(
        [
            ActionEnum.PLAY_HIGHEST_NUM,
        ]
    )
    action_space_two_legal_lead = CEOActionSpace(
        [
            ActionEnum.PLAY_HIGHEST_NUM,
            ActionEnum.PLAY_LOWEST_WITHOUT_BREAK_NUM,
        ]
    )
    action_space_play = CEOActionSpace(
        [
            ActionEnum.PLAY_HIGHEST_NUM,
            ActionEnum.PLAY_SECOND_LOWEST_WITHOUT_BREAK_NUM,
            ActionEnum.PLAY_LOWEST_WITHOUT_BREAK_NUM,
            ActionEnum.PASS_ON_TRICK_NUM,
        ]
    )
    action_space_one_legal_play = CEOActionSpace(
        [
            ActionEnum.PLAY_HIGHEST_NUM,
            ActionEnum.PASS_ON_TRICK_NUM,
        ]
    )
    action_space_two_legal_play = CEOActionSpace(
        [
            ActionEnum.PLAY_HIGHEST_NUM,
            ActionEnum.PLAY_LOWEST_WITHOUT_BREAK_NUM,
            ActionEnum.PASS_ON_TRICK_NUM,
        ]
    )

    _cur_hand: Hand
    _cur_trick_count: int
    _cur_trick_value: CardValue

    _simple_behavior_base: SimpleBehaviorBase

    observation_factory: ObservationFactory

    def __init__(
        self,
        num_players=6,
        behaviors=[],
        hands=[],
        listener=EventListenerInterface(),
        skip_passing=False,
    ):
        self.num_players = num_players
        self._skip_passing = skip_passing

        if listener is None:
            self._listener = EventListenerInterface()
        else:
            self._listener = listener

        self._actions = Actions()

        self._hands = hands

        self._players = []
        self._players.append(Player("RL", RLBehavior()))
        for i in range(num_players - 1):
            name = "Basic" + str(i + 1)

            if i >= len(behaviors):
                self._players.append(Player(name, BasicBehavior()))
            else:
                self._players.append(Player(name, behaviors[i]))

        self.observation_factory = ObservationFactory(num_players)
        self._observation_dimension = self.observation_factory.observation_dimension

        self.observation_space = Box(
            low=np.array([0] * self._observation_dimension),
            high=np.array([13] * self._observation_dimension),
            dtype=np.int32,
        )

        self.action_space = self.action_space_lead
        self.max_action_value = len(ActionEnum)

        self._simple_behavior_base = SimpleBehaviorBase()

    def reset(self, hands: list[Hand] = None):
        self._listener.start_round(self._players)
        self.action_space = self.action_space_lead

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

        gen_tuple = next(self._gen)
        assert gen_tuple[0] == "lead"

        return self._make_observation(gen_tuple)

    def step_full_action(self, full_action):
        self.step(self.action_space.find_full_action(full_action))

    def step(self, action):
        assert (
            isinstance(action, int) or isinstance(action, np.int32) or isinstance(action, np.int64)
        )

        assert action < self.action_space.n

        full_action = self.action_space.actions[action]

        cv = self._actions.play(
            self._cur_hand, self._cur_trick_value, self._cur_trick_count, full_action
        )

        if cv is None and self._cur_trick_value is None:
            print(
                "Action",
                action,
                "full action",
                full_action,
                "returned None to play on trick. Hand",
                self._cur_hand,
            )
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

        # Count the number of playable card values.
        playable_card_values = 0
        for cv in range(13):
            if cur_hand.count(CardValue(cv)) > 0:
                playable_card_values += 1

        # Set up the action space
        if playable_card_values == 1:
            self.action_space = self.action_space_one_legal_lead
        elif playable_card_values == 2:
            self.action_space = self.action_space_two_legal_lead
        else:
            self.action_space = self.action_space_lead

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

        playable_card_values = len(
            self._simple_behavior_base.get_playable_cards(cur_hand, cur_card_value, cur_card_count)
        )

        # See if we must pass, i.e., there is no choice of action
        if playable_card_values == 0:
            return None

        # Set up the action space
        if playable_card_values == 1:
            self.action_space = self.action_space_one_legal_play
        elif playable_card_values == 2:
            self.action_space = self.action_space_two_legal_play
        else:
            self.action_space = self.action_space_play

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
