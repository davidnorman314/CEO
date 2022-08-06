import numpy as np

from gym.spaces import Box, Discrete

from gym_ceo.envs.seat_ceo_env import SeatCEOEnv
from gym_ceo.envs.observation import Observation


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

    def notify_other_features(self, other_features: list):
        pass

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


class HighestCard:
    """Feature giving the value of highest card in the hand. The feature value is zero if
    the highest value in the hand equals min_card_value. The maximum possible value
    is 12 - min_card_value.
    """

    dim = 1
    max_value: int

    min_card_value: int

    def __init__(self, full_env: SeatCEOEnv, *, min_card_value: int):
        self.min_card_value = min_card_value
        self.max_value = 12 - self.min_card_value

    def notify_other_features(self, other_features: list):
        pass

    def calc(
        self,
        full_obs: Observation,
        dest_obs: np.array,
        dest_start_index: int,
        info: dict,
    ):
        # Find the highest value in the hand
        for highest_value in range(12, -1, -1):
            if full_obs.get_card_count(highest_value) > 0:
                break

        feature_value = highest_value - self.min_card_value
        if feature_value < 0:
            feature_value = 0

        dest_obs[dest_start_index] = feature_value
        info[f"HighestCard value={highest_value}"] = feature_value


class HandCardCount:
    """
    Feature giving the number of cards of a given value in the hand
    """

    dim = 1
    max_value = 4
    card_value_index: int

    def __init__(self, full_env: SeatCEOEnv, *, card_value_index: int):
        self.card_value_index = card_value_index

    def notify_other_features(self, other_features: list):
        pass

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


class HandCardCountRelative:
    """
    Feature giving the number of cards of a given value in the hand, relative to the
    highest card. So it can give the number of highest cards, second highest, etc.
    """

    dim = 1
    max_value: int
    relative_card_value: int

    def __init__(self, full_env: SeatCEOEnv, *, relative_card_value: int, max_value):
        """Creates a feature giving the number of cards with value highest_value +
        relative_card_value. The argument relative_card_value must be zero or negative."""

        assert relative_card_value <= 0

        self.relative_card_value = relative_card_value
        self.max_value = max_value

    def notify_other_features(self, other_features: list):
        pass

    def calc(
        self,
        full_obs: Observation,
        dest_obs: np.array,
        dest_start_index: int,
        info: dict,
    ):
        # Find the highest value in the hand
        for highest_value in range(12, -1, -1):
            if full_obs.get_card_count(highest_value) > 0:
                break

        value = highest_value + self.relative_card_value

        if value >= 0:
            card_count = full_obs.get_card_count(value)
        else:
            card_count = 0

        feature_value = min(card_count, self.max_value)
        dest_obs[dest_start_index] = feature_value
        info[
            f"HandCardCountRelative(relative={self.relative_card_value}, value={value}, highest={highest_value})"
        ] = feature_value


class OtherPlayerHandCount:
    """
    Feature giving the number of cards in another players hand
    """

    dim = 1

    other_player_index: int
    max_value: int

    def __init__(self, full_env: SeatCEOEnv, *, other_player_index: int, max_value: int):
        self.other_player_index = other_player_index
        self.max_value = max_value

    def notify_other_features(self, other_features: list):
        pass

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

    def notify_other_features(self, other_features: list):
        pass

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
    Feature giving the number of doubles in the hand below a certain card value
    """

    dim = 1
    max_value = 3

    _threshold: int

    def __init__(self, full_env: SeatCEOEnv, *, threshold: int):
        self._threshold = threshold

    def notify_other_features(self, other_features: list):
        pass

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
    Feature giving the number of triples and larger groups in the hand below a certain card value.
    """

    dim = 1
    max_value = 2

    _threshold: int

    def __init__(self, full_env: SeatCEOEnv, *, threshold: int):
        self._threshold = threshold

    def notify_other_features(self, other_features: list):
        pass

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


class SinglesUnderValueCountRelative:
    """Feature giving the number of singles in the hand below a certain card value that is relative
    to the highest card in the hand.
    """

    dim = 1
    max_value: int

    _relative_threshold: int

    def __init__(self, full_env: SeatCEOEnv, *, relative_threshold: int, max_value: int):
        self._relative_threshold = relative_threshold
        self.max_value = max_value

    def notify_other_features(self, other_features: list):
        pass

    def calc(
        self,
        full_obs: Observation,
        dest_obs: np.array,
        dest_start_index: int,
        info: dict,
    ):
        # Find the highest value in the hand
        for highest_value in range(12, -1, -1):
            if full_obs.get_card_count(highest_value) > 0:
                break

        # Count the singles
        single_count = 0
        for i in range(highest_value - self._relative_threshold):
            card_count = full_obs.get_card_count(i)
            if card_count == 1:
                single_count += 1

        dest_obs[dest_start_index] = min(single_count, self.max_value)
        info["SinglesUnderValueCountRelative"] = dest_obs[dest_start_index]


class DoublesUnderValueCountRelative:
    """Feature giving the number of doubles in the hand below a certain card value
    that is relative to the highest card in the hand.
    """

    dim = 1
    max_value: int

    _relative_threshold: int

    def __init__(self, full_env: SeatCEOEnv, *, relative_threshold: int, max_value: int):
        self._relative_threshold = relative_threshold
        self.max_value = max_value

    def notify_other_features(self, other_features: list):
        pass

    def calc(
        self,
        full_obs: Observation,
        dest_obs: np.array,
        dest_start_index: int,
        info: dict,
    ):
        # Find the highest value in the hand
        for highest_value in range(12, -1, -1):
            if full_obs.get_card_count(highest_value) > 0:
                break

        # Count the doubles
        double_count = 0
        for i in range(highest_value - self._relative_threshold):
            card_count = full_obs.get_card_count(i)
            if card_count == 2:
                double_count += 1

        dest_obs[dest_start_index] = min(double_count, self.max_value)
        info["DoublesUnderValueCountRelative"] = dest_obs[dest_start_index]


class TriplesUnderValueCountRelative:
    """Feature giving the number of triples and larger groups in the hand below a certain card value
    that is relative to the highest card in the hand.
    """

    dim = 1
    max_value: int

    _relative_threshold: int

    def __init__(self, full_env: SeatCEOEnv, *, relative_threshold: int, max_value: int):
        self._relative_threshold = relative_threshold
        self.max_value = max_value

    def notify_other_features(self, other_features: list):
        pass

    def calc(
        self,
        full_obs: Observation,
        dest_obs: np.array,
        dest_start_index: int,
        info: dict,
    ):
        # Find the highest value in the hand
        for highest_value in range(12, -1, -1):
            if full_obs.get_card_count(highest_value) > 0:
                break

        # Count the groups
        triple_count = 0
        for i in range(highest_value - self._relative_threshold):
            card_count = full_obs.get_card_count(i)
            if card_count >= 3:
                triple_count += 1

        dest_obs[dest_start_index] = min(triple_count, self.max_value)
        info["TriplesUnderValueCountRelative"] = dest_obs[dest_start_index]


class ValuesInRangeCount:
    """
    Feature giving the number of values in the hand in the range [range_begin, range_end)
    """

    dim = 1
    max_value: int

    _range_begin: int
    _range_end: int

    def __init__(self, full_env: SeatCEOEnv, *, range_begin: int, range_end: int, max_value: int):
        assert range_begin < range_end

        self._range_begin = range_begin
        self._range_end = range_end
        self.max_value = max_value

    def notify_other_features(self, other_features: list):
        pass

    def calc(
        self,
        full_obs: Observation,
        dest_obs: np.array,
        dest_start_index: int,
        info: dict,
    ):
        count = 0
        for i in range(self._range_begin, self._range_end):
            card_count = full_obs.get_card_count(i)
            if card_count > 0:
                count += 1

        dest_obs[dest_start_index] = min(count, self.max_value)
        info["ValuesInRange({self._range_begin},{self._range_end},{self.max_value})"] = dest_obs[
            dest_start_index
        ]


class HandSummary:
    """Feature giving a summary of the hand."""

    dim: int
    max_value: list[int]

    _high_card_exact_count: int
    _high_card_obs_max: int

    _high_card_value_index: int
    """The index of value of the highest card in the hand"""
    _high_card_value_offset: int
    """The offest for the highest card in the hand. The value of the highest card is the observation
    plus this offset"""

    _high_card_index_start: int
    _bucket_index_start: int
    _hand_count_index: int

    def __init__(
        self,
        full_env: SeatCEOEnv,
        *,
        high_card_exact_count: int,
        high_card_obs_max: int,
        bucket_count: int,
        bucket_obs_max: int,
        include_hand_count: bool,
    ):
        self._high_card_exact_count = high_card_exact_count
        self._high_card_obs_max = high_card_obs_max
        self._bucket_count = bucket_count
        self._bucket_obs_max = bucket_obs_max

        self.dim = 0
        self.max_value = []

        self._high_card_value_index = self.dim
        self.dim += 1
        self._high_card_value_offset = high_card_exact_count + bucket_count - 1
        self.max_value = self.max_value + [13 - self._high_card_value_offset]

        self._high_card_index_start = self.dim
        self.dim += high_card_exact_count
        self.max_value = self.max_value + [high_card_obs_max] * high_card_exact_count

        self._bucket_index_start = self.dim
        self.dim += bucket_count
        self.max_value = self.max_value + [bucket_obs_max] * bucket_count

        if include_hand_count:
            self._hand_count_index = self.dim
            self.dim += 1
            self.max_value = self.max_value + [12]
        else:
            self._hand_count_index = None

    def notify_other_features(self, other_features: list):
        pass

    def get_high_card(
        self,
        obs: np.array,
        dest_start_index: int,
    ):
        """Returns the highest card in the hand."""
        return int(
            obs[self._high_card_value_index + dest_start_index] + self._high_card_value_offset
        )

    def get_high_card_count(self, obs: np.array, dest_start_index: int, i: int):
        """Returns the number of cards with the (i+1)th highest value."""
        assert i < self._high_card_exact_count
        return int(obs[self._high_card_index_start + i + dest_start_index])

    def get_bucket_card_count(self, obs: np.array, dest_start_index: int, i: int):
        """Returns the number of cards in the (i+1)th bucket."""
        assert i < self._bucket_count
        return int(obs[self._bucket_index_start + i + dest_start_index])

    def get_hand_card_count(
        self,
        obs: np.array,
        dest_start_index: int,
    ):
        """Returns the number of cards in the (i+1)th bucket."""
        return int(obs[self._hand_count_index + dest_start_index]) + 1

    def log_features(self, obs: np.array, dest_start_index: int, prefix: str):
        highest_value = self.get_high_card(obs, dest_start_index)
        print(prefix, "High card ", highest_value, sep="")
        for i in range(0, self._high_card_exact_count):
            print(
                prefix,
                "High card count (",
                i,
                ") = ",
                self.get_high_card_count(obs, dest_start_index, i),
                sep="",
            )

        i = 0
        for bucket_start, bucket_end in self._get_buckets(highest_value):
            print(
                prefix,
                "Bucket card count ",
                i,
                " [",
                bucket_start,
                ",",
                bucket_end,
                ") = ",
                self.get_bucket_card_count(obs, dest_start_index, i),
                sep="",
            )

            i += 1

        assert i == self._bucket_count

        print(prefix, "Hand card count ", self.get_hand_card_count(obs, dest_start_index), sep="")

    def _get_buckets(self, highest_value: int) -> tuple[int, int]:
        # Calculate the bucket parameters
        max_bucket_card_value = highest_value - self._high_card_exact_count
        base_bucket_width = (max_bucket_card_value + 1) // self._bucket_count
        extra_values = (max_bucket_card_value + 1) - base_bucket_width * self._bucket_count

        bucket_start = 0
        for i in range(self._bucket_count):
            bucket_end = bucket_start + base_bucket_width
            if i < extra_values:
                bucket_end += 1

            yield bucket_start, bucket_end

            bucket_start = bucket_end

        assert bucket_start - 1 == highest_value - self._high_card_exact_count

    def calc(
        self,
        full_obs: Observation,
        dest_obs: np.array,
        dest_start_index: int,
        info: dict,
    ):
        # find the highest value in the hand
        for highest_value in range(12, -1, -1):
            if full_obs.get_card_count(highest_value) > 0:
                break

        # Save the highest value as a feature. The minimum value for the feature is zero, which
        # corresponds to a highest value where there is one card value for each exact card count
        # and each bucket (of size one).
        if highest_value < self._high_card_value_offset:
            highest_value = self._high_card_value_offset
        feature_value = highest_value - self._high_card_value_offset
        dest_obs[self._high_card_value_index + dest_start_index] = feature_value

        # Save the counts of the high cards that we track exactly.
        for i in range(self._high_card_exact_count):
            card_value = highest_value - i
            card_count = full_obs.get_card_count(card_value)

            feature_value = min(card_count, self._high_card_obs_max)
            dest_obs[self._high_card_index_start + i + dest_start_index] = feature_value

        # Save the counts of cards in each bucket
        i = 0
        for bucket_start, bucket_end in self._get_buckets(highest_value):
            card_count = 0
            for card_value in range(bucket_start, bucket_end):
                card_count += full_obs.get_card_count(card_value)

            feature_value = min(card_count, self._bucket_obs_max)
            dest_obs[self._bucket_index_start + i + dest_start_index] = feature_value

            i += 1

        # Save the number of cards in the hand. Note that the hand must have at least
        # one card.
        if self._hand_count_index:
            hand_card_count = 0
            for i in range(0, 13):
                hand_card_count += full_obs.get_card_count(i)
            feature_value = hand_card_count - 1
            dest_obs[self._hand_count_index + dest_start_index] = feature_value
        info[f"HandSummary(bucket_start={bucket_start})"] = feature_value


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

    def notify_other_features(self, other_features: list):
        pass

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

    def notify_other_features(self, other_features: list):
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

    def notify_other_features(self, other_features: list):
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


class WillWinTrick_AfterState:
    """
    Boolean-valued feature that is true if and only if the agent must win the trick. This only
    works if the state is an afterstate.
    """

    _notify_called: bool

    _max_watch_other: int

    dim = 1
    max_value = 1
    num_players: int

    def __init__(self, full_env: SeatCEOEnv):
        self.num_players = full_env.num_players
        self._notify_called = False

    def notify_other_features(self, other_features: list):
        self._notify_called = True

        other_players = []
        for other in other_features:
            if isinstance(other, OtherPlayerHandCount):
                other_players.append(other.other_player_index)

        other_players.sort()
        # test_list = range(1, len(other_players) + 1)
        # print("eq", test_list == other_players)

        if len(other_players) == 0:
            self._max_watch_other = 0
        elif other_players == list(range(1, len(other_players) + 1)):
            self._max_watch_other = other_players[-1]
        else:
            self._max_watch_other = 0

    def calc(
        self,
        full_obs: Observation,
        dest_obs: np.array,
        dest_start_index: int,
        info: dict,
    ):
        assert self._notify_called

        trick_start_player = full_obs.get_starting_player().astype(int)
        trick_last_player = full_obs.get_last_player().astype(int)
        cur_value = full_obs.get_cur_trick_value().astype(int)
        trick_size = full_obs.get_cur_trick_count().astype(int)

        # We will win if the agent led an ace
        if cur_value == 12 and trick_last_player == 0:
            dest_obs[dest_start_index] = 1
            info["WillWinTrick"] = "played ace"

            return

        # We will win the trick if the person after us led and we play on the trick
        if trick_start_player == 1 and trick_last_player == 0:
            dest_obs[dest_start_index] = 1
            info["WillWinTrick"] = "last player"

            return

        # If we played on the trick and if we have features for the players yet to play,
        # we can check their hand counts. Note that this assumes that the feature has values large
        # enough to differentiate the current trick size.
        if (
            trick_start_player != 0
            and trick_last_player == 0
            and trick_start_player <= self._max_watch_other + 1
        ):
            all_hands_small = True
            for other_player_index in range(1, trick_start_player):
                other_card_count = full_obs.get_other_player_card_count(other_player_index - 1)
                if other_card_count >= trick_size:
                    all_hands_small = False
                    break

            if all_hands_small:
                dest_obs[dest_start_index] = 1
                info[
                    "WillWinTrick"
                ] = f"all small start_player={trick_start_player} max_watch={self._max_watch_other}"

                return

        dest_obs[dest_start_index] = 0
        info["WillWinTrick"] = "May not win"

    def log_features(self, obs: np.array, dest_start_index: int, prefix: str):
        value = obs[dest_start_index]
        print(
            prefix,
            "WillWin" if value > 0 else "MayNotWin",
            sep="",
        )


class FeatureObservationFactory:
    """Class that calculates a feature observation from raw observation."""

    full_env: SeatCEOEnv

    feature_defs: list
    """Feature definitions"""

    _feature_calculators: list
    """Feature calculators"""

    observation_space: Box
    observation_dimension: int
    obs_space_possible_values: int

    def __init__(self, full_env: SeatCEOEnv, feature_defs: list):
        self.full_env = full_env
        self._feature_calculators = []
        self.feature_defs = feature_defs

        # Create the features
        for feature_class, kwargs in self.feature_defs:
            class_obj = globals()[feature_class]
            feature = class_obj(full_env, **kwargs)

            self._feature_calculators.append(feature)

        for feature_calculator in self._feature_calculators:
            feature_calculator.notify_other_features(self._feature_calculators)

        # Calculate the observation space
        obs_space_low = []
        obs_space_high = []
        self.obs_space_possible_values = 1
        for calculator in self._feature_calculators:
            max_value = calculator.max_value
            if isinstance(max_value, int):
                assert calculator.dim == 1
                obs_space_low.append(0.0)
                obs_space_high.append(np.float64(max_value))

                self.obs_space_possible_values = self.obs_space_possible_values * (max_value + 1)
            elif isinstance(max_value, list):
                assert calculator.dim == len(max_value)
                for max_value_single in max_value:
                    obs_space_low.append(0)
                    obs_space_high.append(max_value_single)

                    self.obs_space_possible_values = self.obs_space_possible_values * (
                        max_value_single + 1
                    )
            else:
                raise ArgumentError("Unknown max_value type: ", type(max_value))

            print(
                "Calculator",
                type(calculator),
                "dim",
                calculator.dim,
                "max",
                calculator.max_value,
            )

        print("Observation space low:", obs_space_low)
        print("Observation space high:", obs_space_high)
        self.observation_space = Box(
            low=np.array(obs_space_low),
            high=np.array(obs_space_high),
            dtype=np.float64,
        )

        self.observation_dimension = len(obs_space_high)

    def make_feature_observation(self, full_obs_array, info: dict):
        if full_obs_array is None:
            return None

        full_obs = Observation(self.full_env.observation_factory, array=full_obs_array)

        feature_obs_array = np.zeros(self.observation_dimension)

        i = 0
        for calculator in self._feature_calculators:
            ret = calculator.calc(full_obs, feature_obs_array, i, info)
            assert ret is None

            i = i + calculator.dim

        return feature_obs_array

    def log_feature_observation(self, feature_obs_array, prefix: str):
        i = 0
        for calculator in self._feature_calculators:
            log_features = getattr(calculator, "log_features", None)
            if callable(log_features):
                log_features(feature_obs_array, i, prefix)

            i = i + calculator.dim
