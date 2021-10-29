import gym
import numpy as np
from gym import error, spaces, utils
from gym.spaces import Box, Discrete
from gym.utils import seeding

from gym_ceo.envs.seat_ceo_env import SeatCEOEnv
from gym_ceo.envs.actions import Actions

from CEO.cards.round import Round, RoundState
from CEO.cards.eventlistener import EventListenerInterface
from CEO.cards.deck import Deck
from CEO.cards.hand import Hand, CardValue
from CEO.cards.simplebehavior import BasicBehavior
from CEO.cards.player import Player


class BottomHalfTableMinCards:
    """
    Feature giving the minimum number of cards for players in the bottom half of the table.
    """

    dim = 1
    max_value = 5

    _start_check_index: int
    _end_check_index: int

    def __init__(self, full_env: SeatCEOEnv):
        # Note that the agent's hand card count is not included in the observation.
        self._start_check_index = (
            full_env.num_players // 2 - 1 + full_env.obs_index_other_player_card_count
        )
        self._end_check_index = (
            full_env.num_players - 1 + full_env.obs_index_other_player_card_count
        )
        print(full_env.obs_index_other_player_card_count)

    def calc(self, full_obs: np.array, dest_obs: np.array, dest_start_index: int, info: dict):
        feature_value = self.max_value
        for i in range(self._start_check_index, self._end_check_index):
            feature_value = min(feature_value, full_obs[i])

        # If another player is out, then CEO goes to the bottom of the table
        assert feature_value != 0

        dest_obs[dest_start_index] = feature_value
        info["BottomHalfTableMinCards"] = feature_value


class HandCardCount:
    """
    Feature giving the number of cards of a given value in the hand
    """

    dim = 1
    max_value = 4
    full_obs_index: int
    card_value_index: int

    def __init__(self, full_env: SeatCEOEnv, card_value_index: int):
        self.full_obs_index = full_env.obs_index_hand_cards + card_value_index
        self.card_value_index = card_value_index

    def calc(self, full_obs: np.array, dest_obs: np.array, dest_start_index: int, info: dict):
        feature_value = min(full_obs[self.full_obs_index], self.max_value)
        dest_obs[dest_start_index] = feature_value
        info["HandCardCount " + str(self.card_value_index)] = feature_value


class OtherPlayerHandCount:
    """
    Feature giving the number of cards in another players hand
    """

    dim = 1
    max_value = 5
    full_obs_index: int
    other_player_index: int

    def __init__(self, full_env: SeatCEOEnv, other_player_index: int):
        self.other_player_index = other_player_index
        self.full_obs_index = full_env.obs_index_other_player_card_count + other_player_index

    def calc(self, full_obs: np.array, dest_obs: np.array, dest_start_index: int, info: dict):
        feature_value = min(full_obs[self.full_obs_index], self.max_value)
        dest_obs[dest_start_index] = feature_value
        info["OtherPlayerHandCount " + str(self.other_player_index)] = feature_value

        # If another player is out, then CEO goes to the bottom of the table
        if feature_value == 0:
            print(full_obs)
            print("full obs index", self.full_obs_index)
            print("other player index", self.other_player_index)
        assert feature_value != 0


class SinglesUnderValueCount:
    """
    Feature giving the number of singles in the hand below a certain card value
    """

    dim = 1
    max_value = 3

    _start_check_index: int
    _end_check_index: int

    def __init__(self, full_env: SeatCEOEnv, threshold: int):
        self._start_check_index = full_env.obs_index_hand_cards
        self._end_check_index = full_env.obs_index_hand_cards + threshold

    def calc(self, full_obs: np.array, dest_obs: np.array, dest_start_index: int, info: dict):
        single_count = 0
        for i in range(self._start_check_index, self._end_check_index):
            card_count = full_obs[i]
            if card_count == 1:
                single_count += 1

        dest_obs[dest_start_index] = min(single_count, self.max_value)
        info["SinglesUnderValueCount"] = dest_obs[dest_start_index]


class DoublesUnderValueCount:
    """
    Feature giving the number of singles in the hand below a certain card value
    """

    dim = 1
    max_value = 3

    _start_check_index: int
    _end_check_index: int

    def __init__(self, full_env: SeatCEOEnv, threshold: int):
        self._start_check_index = full_env.obs_index_hand_cards
        self._end_check_index = full_env.obs_index_hand_cards + threshold

    def calc(self, full_obs: np.array, dest_obs: np.array, dest_start_index: int, info: dict):
        double_count = 0
        for i in range(self._start_check_index, self._end_check_index):
            card_count = full_obs[i]
            if card_count == 2:
                double_count += 1

        dest_obs[dest_start_index] = min(double_count, self.max_value)
        info["DoublesUnderValueCount"] = dest_obs[dest_start_index]


class TriplesUnderValueCount:
    """
    Feature giving the number of singles in the hand below a certain card value
    """

    dim = 1
    max_value = 2

    _start_check_index: int
    _end_check_index: int

    def __init__(self, full_env: SeatCEOEnv, threshold: int):
        self._start_check_index = full_env.obs_index_hand_cards
        self._end_check_index = full_env.obs_index_hand_cards + threshold

    def calc(self, full_obs: np.array, dest_obs: np.array, dest_start_index: int, info: dict):
        triple_count = 0
        for i in range(self._start_check_index, self._end_check_index):
            card_count = full_obs[i]
            if card_count >= 3:
                triple_count += 1

        dest_obs[dest_start_index] = min(triple_count, self.max_value)
        info["TriplesUnderValueCount"] = dest_obs[dest_start_index]


class TrickPosition:
    """
    Feature giving the position the player is for the trick: lead, in the middle, or
    last one to play.
    """

    dim = 1
    max_value = 2
    full_start_player_index: int
    num_player: int

    def __init__(self, full_env: SeatCEOEnv):
        self.full_start_player_index = full_env.obs_index_start_player
        self.num_player = full_env.num_players

    def calc(self, full_obs: np.array, dest_obs: np.array, dest_start_index: int, info: dict):
        start_player = full_obs[self.full_start_player_index]

        if start_player == 0:
            # The agent leads
            dest_obs[dest_start_index] = 0
            info["Trick position"] = "lead"
        elif start_player == 1:
            # The agent is the last player on the trick
            dest_obs[dest_start_index] = 2
            info["Trick position"] = "last"
        else:
            dest_obs[dest_start_index] = 1
            info["Trick position"] = "not lead and not last"


class CurTrickValue:
    """
    Feature giving the current trick's value.
    0 - Below all cards in hand. This also means that the player should lead.
    1 - Above one card value in hand
    2 - Above two card values in hand
    3 - Above three card values in hand
    4 - Other: Above four or more cards in the hand and less than three or more.
    5 - Below two card values in hand (there are two values that can be played.)
    6 - Below one card value in hand (there is one value that can be played.)
    """

    dim = 1
    max_value = 6

    _obs_index_cur_trick_count: int
    _obs_index_cur_trick_value: int
    _obs_index_hand_cards: int

    def __init__(self, full_env: SeatCEOEnv):
        self._obs_index_cur_trick_count = full_env.obs_index_cur_trick_count
        self._obs_index_cur_trick_value = full_env.obs_index_cur_trick_value
        self._obs_index_hand_cards = full_env.obs_index_hand_cards
        pass

    def calc(self, full_obs: np.array, dest_obs: np.array, dest_start_index: int, info: dict):
        if full_obs[self._obs_index_cur_trick_count] == 0:
            # We should lead
            dest_obs[dest_start_index] = 0
            info["CurTrickValue"] = "lead"
            return
        else:
            cur_trick_value = full_obs[self._obs_index_cur_trick_value]
            hand_below_count = 0
            hand_above_count = 0
            for i in range(13):
                count = full_obs[self._obs_index_hand_cards + i]
                if count > 0:
                    if cur_trick_value >= i:
                        hand_below_count += 1
                    else:
                        hand_above_count += 1

            assert hand_above_count != 0

            if hand_below_count <= 3:
                dest_obs[dest_start_index] = hand_below_count
                info["CurTrickValue"] = "hand_below_count " + str(hand_below_count)
            elif hand_above_count <= 2:
                dest_obs[dest_start_index] = self.max_value - hand_above_count + 1
                info["CurTrickValue"] = "hand_below_count " + str(dest_obs[dest_start_index])
            else:
                dest_obs[dest_start_index] = 4
                info["CurTrickValue"] = "Other"


class CurTrickCount:
    """
    Feature giving the number of cards in the current trick's.
    0 - The trick consists of singles. This also means that we should lead.
    1 - The trick consists of doubles
    2 - The trick consists of triples
    3 - The trick consists of quadruples or larger.
    """

    dim = 1
    max_value = 3

    _obs_index_cur_trick_count: int

    def __init__(self, full_env: SeatCEOEnv):
        self._obs_index_cur_trick_count = full_env.obs_index_cur_trick_count
        pass

    def calc(self, full_obs: np.array, dest_obs: np.array, dest_start_index: int, info: dict):
        cur_trick_cards = full_obs[self._obs_index_cur_trick_count]

        # See if we should lead
        if cur_trick_cards == 0:
            dest_obs[dest_start_index] = 0
            info["CurTrickCount"] = "lead"
            return

        obs = cur_trick_cards - 1

        if obs > self.max_value:
            obs = self.max_value

        dest_obs[dest_start_index] = obs
        info["CurTrickCount"] = obs


class SeatCEOFeaturesEnv(gym.Env):
    """
    Environment for a player in the CEO seat. This environment reduces the observation space
    to a set of features.
    """

    metadata = {"render.modes": ["human"]}

    full_env: SeatCEOEnv

    observation_space: Box
    action_space: Discrete
    max_action_value: int

    # Objects to calculate the features
    _feature_calculators: list

    _observation_dimension: int

    def __init__(self, full_env: SeatCEOEnv):
        self.full_env = full_env
        self.action_space = full_env.action_space
        self.max_action_value = full_env.max_action_value

        self._feature_calculators = []

        half_players = full_env.num_players // 2
        for i in range(half_players - 1):
            feature = OtherPlayerHandCount(full_env, i)
            self._feature_calculators.append(feature)

        # self._feature_calculators.append(BottomHalfTableMinCards(full_env))

        min_card_exact_feature = 9
        for i in range(min_card_exact_feature, 13):
            self._feature_calculators.append(HandCardCount(full_env, i))
        self._feature_calculators.append(SinglesUnderValueCount(full_env, min_card_exact_feature))
        self._feature_calculators.append(DoublesUnderValueCount(full_env, min_card_exact_feature))
        self._feature_calculators.append(TriplesUnderValueCount(full_env, min_card_exact_feature))

        self._feature_calculators.append(TrickPosition(full_env))
        self._feature_calculators.append(CurTrickValue(full_env))
        self._feature_calculators.append(CurTrickCount(full_env))

        # Calculate the observation space
        obs_space_low = []
        obs_space_high = []
        obs_space_possible_values = 1
        for calculator in self._feature_calculators:
            for i in range(calculator.dim):
                obs_space_low.append(0)
                obs_space_high.append(calculator.max_value)
                obs_space_possible_values = obs_space_possible_values * (calculator.max_value + 1)
            print(
                "Calculator", type(calculator), "dim", calculator.dim, "max", calculator.max_value
            )
        self.observation_space = Box(
            low=np.array(obs_space_low),
            high=np.array(obs_space_high),
            dtype=np.int32,
        )

        self._observation_dimension = len(obs_space_high)
        print(
            "SeatCEOFeatures env has",
            self._observation_dimension,
            "dimensional observation space with ",
            obs_space_possible_values,
            "possible values",
        )

    def reset(self, hands: list[Hand] = None):
        full_obs = self.full_env.reset(hands)
        self.action_space = self.full_env.action_space

        info = dict()
        return self._make_observation(full_obs, info)

    def step(self, action):
        full_obs, reward, done, info = self.full_env.step(action)
        self.action_space = self.full_env.action_space

        obs = self._make_observation(full_obs, info)
        return obs, reward, done, info

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def _make_observation(self, full_obs, info: dict):
        if full_obs is None:
            return None

        obs = np.zeros(self._observation_dimension)

        i = 0
        for calculator in self._feature_calculators:
            ret = calculator.calc(full_obs, obs, i, info)
            assert ret is None

            i = i + calculator.dim

        return obs
