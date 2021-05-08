import array
from CEO.cards.hand import *
from CEO.cards.player import *
from CEO.cards.eventlistener import EventListenerInterface


class BehaviorStatistics:
    """
    Class that keeps track of statistics for a single behavior
    """

    total_players: int
    # start_position_count: array.array[int]
    # end_position_count: array.array[int]
    # move_up_delta: array.array[int]
    # move_down_delta: array.array[int]
    start_position_count: list[int]
    end_position_count: list[int]
    move_up_delta: list[int]
    move_down_delta: list[int]

    def __init__(self, total_players: int):
        self.total_players = total_players

        zero_list = [0] * total_players

        self.start_position_count = array.array("i", zero_list)
        self.end_position_count = array.array("i", zero_list)
        self.move_up_delta = array.array("i", zero_list)
        self.move_down_delta = array.array("i", zero_list)

    def add_round_result(self, start_position: int, end_position: int):
        self.start_position_count[start_position] += 1
        self.end_position_count[end_position] += 1


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
