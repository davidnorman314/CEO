import gym
import numpy as np
from gym import error, spaces, utils
from gym.spaces import Box, Discrete
from gym.utils import seeding

from gym_ceo.envs.seat_ceo_env import SeatCEOEnv
from gym_ceo.envs.actions import Actions
from gym_ceo.envs.observation import ObservationFactory, Observation

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
        self._start_check_index = full_env.num_players // 2 - 1
        self._end_check_index = full_env.num_players - 1

    def calc(
        self,
        full_obs: Observation,
        dest_obs: np.array,
        dest_start_index: int,
        info: dict,
    ):
        feature_value = self.max_value
        for i in range(self._start_check_index, self._end_check_index):
            card_count = full_obs.get_other_player_card_count(i)
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
    card_value_index: int

    def __init__(self, full_env: SeatCEOEnv, *, card_value_index: int):
        self.card_value_index = card_value_index

    def calc(
        self,
        full_obs: Observation,
        dest_obs: np.array,
        dest_start_index: int,
        info: dict,
    ):
        card_count = full_obs.get_card_count(self.card_value_index)
        feature_value = min(card_count, self.max_value)
        dest_obs[dest_start_index] = feature_value
        info["HandCardCount " + str(self.card_value_index)] = feature_value


class OtherPlayerHandCount:
    """
    Feature giving the number of cards in another players hand
    """

    dim = 1
    max_value = 5

    other_player_index: int

    def __init__(self, full_env: SeatCEOEnv, *, other_player_index: int):
        self.other_player_index = other_player_index

    def calc(
        self,
        full_obs: Observation,
        dest_obs: np.array,
        dest_start_index: int,
        info: dict,
    ):
        other_player_card_count = full_obs.get_other_player_card_count(self.other_player_index)
        feature_value = min(other_player_card_count, self.max_value)
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

    _threshold: int

    def __init__(self, full_env: SeatCEOEnv, *, threshold: int):
        self._threshold = threshold

    def calc(
        self,
        full_obs: Observation,
        dest_obs: np.array,
        dest_start_index: int,
        info: dict,
    ):
        single_count = 0
        for i in range(self._threshold):
            card_count = full_obs.get_card_count(i)
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

    _threshold: int

    def __init__(self, full_env: SeatCEOEnv, *, threshold: int):
        self._threshold = threshold

    def calc(
        self,
        full_obs: Observation,
        dest_obs: np.array,
        dest_start_index: int,
        info: dict,
    ):
        double_count = 0
        for i in range(self._threshold):
            card_count = full_obs.get_card_count(i)
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

    _threshold: int

    def __init__(self, full_env: SeatCEOEnv, *, threshold: int):
        self._threshold = threshold

    def calc(
        self,
        full_obs: Observation,
        dest_obs: np.array,
        dest_start_index: int,
        info: dict,
    ):
        triple_count = 0
        for i in range(self._threshold):
            card_count = full_obs.get_card_count(i)
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
    num_player: int

    def __init__(self, full_env: SeatCEOEnv):
        self.num_player = full_env.num_players

    def calc(
        self,
        full_obs: Observation,
        dest_obs: np.array,
        dest_start_index: int,
        info: dict,
    ):
        start_player = full_obs.get_starting_player()

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
    Feature giving the current trick's value relative to the cards in the hand.
    0 - Below all cards in hand and the hand's lowest value can be played without
        breaking a set.
        This also means that the player should lead.
        Note that if doubles are played, then the trick value might be less than
        all cards in the hand, but the lowest value can't be played if it is
        a single.
    1 - Below all cards in hand.
    2 - Above one card value in hand
    3 - Above two card values in hand
    4 - Above three card values in hand
    5 - Other: Above four or more cards in the hand and less than three or more.
    6 - Below two card values in hand (there are two values that can be played.)
    7 - Below one card value in hand (there is one value that can be played.)
    """

    dim = 1
    max_value = 7

    def __init__(self, full_env: SeatCEOEnv):
        pass

    def calc(
        self,
        full_obs: Observation,
        dest_obs: np.array,
        dest_start_index: int,
        info: dict,
    ):
        cur_trick_count = full_obs.get_cur_trick_count()
        if cur_trick_count == 0:
            # We should lead
            dest_obs[dest_start_index] = 0
            info["CurTrickValue"] = "lead"
            return
        else:
            cur_trick_value = full_obs.get_cur_trick_value()
            hand_below_count = 0
            hand_above_count = 0
            found_lowest = False
            for i in range(13):
                count = full_obs.get_card_count(i)

                if count == 0:
                    continue

                if not found_lowest:
                    found_lowest = True
                    if i >= cur_trick_value and count == cur_trick_count:
                        dest_obs[dest_start_index] = 0
                        info["CurTrickValue"] = "below all and play lowest without breaking"
                        return

                if cur_trick_value >= i:
                    hand_below_count += 1
                else:
                    hand_above_count += 1

            assert hand_above_count != 0

            if hand_below_count <= 3:
                dest_obs[dest_start_index] = hand_below_count + 1
                info["CurTrickValue"] = "hand_below_count " + str(hand_below_count)
            elif hand_above_count <= 2:
                obs = self.max_value - hand_above_count + 1
                dest_obs[dest_start_index] = obs
                info["CurTrickValue"] = "hand_above_count " + str(hand_above_count)
            else:
                dest_obs[dest_start_index] = 5
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

    def __init__(self, full_env: SeatCEOEnv):
        pass

    def calc(
        self,
        full_obs: Observation,
        dest_obs: np.array,
        dest_start_index: int,
        info: dict,
    ):
        cur_trick_count = full_obs.get_cur_trick_count()

        # See if we should lead
        if cur_trick_count == 0:
            dest_obs[dest_start_index] = 0
            info["CurTrickCount"] = "lead"
            return

        obs = cur_trick_count - 1

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

        # Get default feature definitions
        feature_defs = self.get_default_features(full_env)

        # Create the features
        for feature_class, kwargs in feature_defs:
            class_obj = globals()[feature_class]
            feature = class_obj(full_env, **kwargs)

            self._feature_calculators.append(feature)

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
                "Calculator",
                type(calculator),
                "dim",
                calculator.dim,
                "max",
                calculator.max_value,
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

    def get_default_features(self, full_env: SeatCEOEnv):
        feature_defs = []
        half_players = full_env.num_players // 2
        for i in range(half_players - 1):
            feature_params = dict()
            feature_params["other_player_index"] = i
            feature_defs.append(("OtherPlayerHandCount", feature_params))

        if False:
            feature_params = dict()
            feature_defs.append(("BottomHalfTableMinCards", feature_params))

        min_card_exact_feature = 9
        for i in range(min_card_exact_feature, 13):
            feature_params = dict()
            feature_params["card_value_index"] = i
            feature_defs.append(("HandCardCount", feature_params))

        feature_params = dict()
        feature_params["threshold"] = min_card_exact_feature
        feature_defs.append(("SinglesUnderValueCount", feature_params))

        feature_params = dict()
        feature_params["threshold"] = min_card_exact_feature
        feature_defs.append(("DoublesUnderValueCount", feature_params))

        feature_params = dict()
        feature_params["threshold"] = min_card_exact_feature
        feature_defs.append(("TriplesUnderValueCount", feature_params))

        feature_params = dict()
        feature_defs.append(("TrickPosition", feature_params))

        feature_params = dict()
        feature_defs.append(("CurTrickValue", feature_params))

        feature_params = dict()
        feature_defs.append(("CurTrickCount", feature_params))

        return feature_defs

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

    def _make_observation(self, full_obs_array, info: dict):
        if full_obs_array is None:
            return None

        full_obs = Observation(self.full_env.observation_factory, array=full_obs_array)

        feature_obs_array = np.zeros(self._observation_dimension)

        i = 0
        for calculator in self._feature_calculators:
            ret = calculator.calc(full_obs, feature_obs_array, i, info)
            assert ret is None

            i = i + calculator.dim

        return feature_obs_array
