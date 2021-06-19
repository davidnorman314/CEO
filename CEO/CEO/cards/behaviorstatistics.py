import array
from CEO.cards.hand import *
from CEO.cards.player import *
from CEO.cards.eventlistener import EventListenerInterface


class BehaviorStatistics:
    """
    Class that keeps track of statistics for a single behavior
    """

    players_with_behavior_count: int
    total_players: int
    # start_position_count: array.array[int]
    # end_position_count: array.array[int]
    # move_up_delta: array.array[int]
    # move_down_delta: array.array[int]
    start_position_count: list[int]
    end_position_count: list[int]
    move_up_delta: list[int]
    move_down_delta: list[int]
    stay_count: int
    start_to_finish: list[list[int]]
    bottom_half_stay: int
    bottom_half_move_up: int
    bottom_half_move_down: int

    _smallest_bottom_half_seat: int

    def __init__(self, total_players: int):
        self.players_with_behavior_count = 1
        self.total_players = total_players
        self.stay_count = 0
        self.bottom_half_stay = 0
        self.bottom_half_move_up = 0
        self.bottom_half_move_down = 0

        self._smallest_bottom_half_seat = total_players - total_players / 2

        zero_list = [0] * total_players

        self.start_position_count = array.array("i", zero_list)
        self.end_position_count = array.array("i", zero_list)
        self.move_up_delta = array.array("i", [0] * total_players)
        self.move_down_delta = array.array("i", [0] * total_players)

        self.start_to_finish = [[0] * total_players for i in range(total_players)]

    def add_another_player_with_behavior(self):
        self.players_with_behavior_count += 1

    def add_round_result(self, start_position: int, end_position: int):
        self.start_position_count[start_position] += 1
        self.end_position_count[end_position] += 1
        self.start_to_finish[start_position][end_position] += 1

        delta = end_position - start_position
        if delta == 0:
            self.stay_count += 1
        elif delta > 0:
            self.move_up_delta[delta] += 1
        elif delta < 0:
            self.move_down_delta[-delta] += 1

        if start_position >= self._smallest_bottom_half_seat:
            if end_position == start_position:
                self.bottom_half_stay += 1
            elif end_position <= start_position:
                self.bottom_half_move_down += 1
            else:
                self.bottom_half_move_up += 1


class BehaviorStatisticsCollector(EventListenerInterface):
    """
    Class that keeps statistics on how each behavior did playing the game.
    """

    _begin_order: list[Player]

    stats: dict[str, BehaviorStatistics]

    def __init__(self, players: list[Player]):
        self.stats = dict()
        count = len(players)
        for player in players:
            behavior_name = player.behavoir.__class__.__name__

            if behavior_name in self.stats:
                self.stats[behavior_name].add_another_player_with_behavior()
            else:
                self.stats[behavior_name] = BehaviorStatistics(count)

    def start_round(self, players: list[Player]):
        self._begin_order = players.copy()

    def end_round(self, next_round_players: list[Player]):
        count = len(self._begin_order)

        for start_position in range(count):
            player = self._begin_order[start_position]
            behavior_name = player.behavoir.__class__.__name__
            behavior_stats = self.stats[behavior_name]

            end_position = [i for i in range(count) if next_round_players[i].name == player.name][0]

            behavior_stats.add_round_result(start_position, end_position)
