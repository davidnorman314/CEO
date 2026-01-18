"""Code to play rounds with various agents and evaluate their performance."""

from ceo.game.eventlistener import (
    EventListenerInterface,
    MultiEventListener,
    PrintAllEventListener,
)
from ceo.game.game import Game
from ceo.game.hand import CardValue, Hand
from ceo.game.player import Player, RoundState
from ceo.game.simplebehavior import BasicBehavior
from ceo.game.winlossstatistics import WinLossStatisticsCollector
from ceo.learning.ppo_agents import process_ppo_agents


class HeuristicMonitorListener(EventListenerInterface):
    """Listener that counts the number of times a behavior uses a specified heuristic,
    e.g., always lead your lowest card.
    """

    class Stats:
        lead_lowest_count: int
        lead_second_lowest_count: int

        lead_lowest_lgdiff_count: int
        lead_second_lgdiff_lowest_count: int

        def __init__(self):
            self.lead_lowest_count = 0
            self.lead_second_lowest_count = 0

            self.lead_lowest_lgdiff_count = 0
            self.lead_second_lgdiff_lowest_count = 0

    stats: dict[int, Stats]

    _seat: int
    _hand: Hand

    def __init__(self, num_players: int):
        self.stats = dict()

        for i in range(num_players):
            self.stats[i] = self.Stats()

    def before_lead(self, index: int, player: Player, hand: Hand, state: RoundState):
        self._seat = index
        self._hand = hand

    def lead(self, cards: CardValue, count: int, index: int, player: Player):
        card_values = self._hand.get_card_values()

        # Don't save statistics if there are one or two values in the hand
        if len(card_values) <= 2:
            return

        # Don't save statistics if there are different numbers of the two cards.
        if card_values[0][1] != card_values[1][1]:
            return

        # Save the count of what was played
        if cards == card_values[0][0]:
            self.stats[index].lead_lowest_count += 1
        elif cards == card_values[1][0]:
            self.stats[index].lead_second_lowest_count += 1

        if card_values[1][0].value - card_values[0][0].value >= 5:
            if cards == card_values[0][0]:
                self.stats[index].lead_lowest_lgdiff_count += 1
            elif cards == card_values[1][0]:
                self.stats[index].lead_second_lgdiff_lowest_count += 1


def evaluate_agents(
    num_players: int,
    num_rounds: int,
    ppo_agents: list[str],
    basic_agent_seats: list[int],
    device: str = None,
    do_print: bool = False,
):
    """Evaluate agents by playing rounds and collecting statistics.

    Args:
        num_players: Number of players in the game.
        num_rounds: Number of rounds to play.
        ppo_agents: List of directories containing trained PPO agents.
        basic_agent_seats: List of seat numbers for BasicBehavior agents.
        device: CUDA device to use (e.g., "cuda" or "cuda:0").
        do_print: If True, print game status instead of collecting stats.

    Returns:
        Tuple of (win_loss_listener, heuristic_monitor, players) if do_print is False,
        otherwise None.
    """
    basic_agent_seats_set = set(basic_agent_seats)

    # Load the agents
    custom_behaviors, custom_behavior_descs = process_ppo_agents(
        ppo_agents, device=device, num_players=num_players
    )

    players = []
    for seat in range(num_players):
        if seat not in custom_behaviors and seat not in basic_agent_seats_set:
            print(f"Basic agents {basic_agent_seats_set}.")
            raise Exception(f"No player specified for seat {seat}.")

        if seat in basic_agent_seats_set:
            assert seat not in custom_behaviors
            name = "Basic" + str(seat)
            players.append(Player(name, BasicBehavior()))

        if seat in custom_behaviors:
            assert seat not in basic_agent_seats_set
            players.append(Player(custom_behavior_descs[seat], custom_behaviors[seat]))

    listeners = []
    win_loss_listener = None
    if do_print:
        listeners.append(PrintAllEventListener())
    else:
        win_loss_listener = WinLossStatisticsCollector(players)
        listeners.append(win_loss_listener)

    heuristic_monitor = HeuristicMonitorListener(num_players)
    listeners.append(heuristic_monitor)

    listener = MultiEventListener(listeners)

    game = Game(players, listener)
    game.play(round_count=num_rounds, do_shuffle=False, reorder_seats=False)

    if do_print:
        return None

    return win_loss_listener, heuristic_monitor, players


def print_statistics(
    win_loss_listener: WinLossStatisticsCollector,
    heuristic_monitor: HeuristicMonitorListener,
    players: list[Player],
    num_players: int,
    num_rounds: int,
):
    """Print win/loss and heuristic statistics."""
    format1 = "{:20}"
    format2 = "{:5.2f}"
    format2a = "{:5.0f}"

    # Print win/loss statistics
    for behavior_name in win_loss_listener.stats:
        print(behavior_name)
        stats = win_loss_listener.stats[behavior_name]

        print(format1.format("Percent in seat:"), end="")
        for i in range(num_players):
            pct = (
                stats.end_position_count[i] / stats.players_with_behavior_count
            ) / num_rounds

            print(format2.format(pct), end="")

        print("")

        total_move_up = 0
        print(format1.format("Move up count:"), end="")
        for i in range(num_players):
            if i == 0:
                continue

            cnt = stats.move_up_delta[i] / stats.players_with_behavior_count
            total_move_up += cnt

            print(format2a.format(cnt), end="")

        print("")

        total_move_down = 0
        print(format1.format("Move down count:"), end="")
        for i in range(num_players):
            if i == 0:
                continue

            cnt = stats.move_down_delta[i] / stats.players_with_behavior_count
            total_move_down += cnt

            print(format2a.format(cnt), end="")

        print("")

        print(format1.format("Up count:"), end="")
        print(format2a.format(total_move_up), end="")
        print("")
        print(format1.format("Stay count:"), end="")
        print(
            format2a.format(stats.stay_count / stats.players_with_behavior_count),
            end="",
        )
        print("")
        print(format1.format("Down count:"), end="")
        print(format2a.format(total_move_down), end="")
        print("")

        print("All changes")
        for i in range(num_players):
            total_for_start_seat = 0
            for j in range(num_players):
                total_for_start_seat += stats.start_to_finish[i][j]

            print(format1.format("Start seat " + str(i)), end="")
            for j in range(num_players):
                if total_for_start_seat > 0:
                    val = stats.start_to_finish[i][j] / total_for_start_seat
                else:
                    val = 0.0
                print(format2.format(val), end="")
            print("")

        # Bottom half stats
        move_up_count = stats.bottom_half_move_up / stats.players_with_behavior_count
        move_down_count = (
            stats.bottom_half_move_down / stats.players_with_behavior_count
        )
        stay_count = stats.bottom_half_stay / stats.players_with_behavior_count
        bottom_half_count = move_up_count + move_down_count + stay_count

        if bottom_half_count > 0:
            print("Bottom half stats:")
            print(format1.format("    Move up:"), end="")
            print(
                format2a.format(move_up_count),
                format2.format(move_up_count / bottom_half_count),
                end="",
            )
            print("")
            print(format1.format("       Stay:"), end="")
            print(
                format2a.format(stay_count),
                format2.format(stay_count / bottom_half_count),
                end="",
            )
            print("")
            print(format1.format("  Move down:"), end="")
            print(
                format2a.format(move_down_count),
                format2.format(move_down_count / bottom_half_count),
                end="",
            )
            print("")

        print("")

    # Print heuristics statistics
    print("Lead lowest heuristics:")
    for seat in range(num_players):
        name = players[seat].name
        lead_lowest_count = heuristic_monitor.stats[seat].lead_lowest_count
        lead_second_lowest_count = heuristic_monitor.stats[
            seat
        ].lead_second_lowest_count

        total = lead_lowest_count + lead_second_lowest_count
        if total == 0:
            print(f"Skipping {name}, since no data")
            continue

        pct_lead_lowest = lead_lowest_count / total

        print(f"{name:20} lead lowest pct {pct_lead_lowest:5.2f} total {total}")

    print("Lead lowest heuristics with large difference:")
    for seat in range(num_players):
        name = players[seat].name
        lead_lowest_count = heuristic_monitor.stats[seat].lead_lowest_lgdiff_count
        lead_second_lowest_count = heuristic_monitor.stats[
            seat
        ].lead_second_lgdiff_lowest_count

        total = lead_lowest_count + lead_second_lowest_count
        if total == 0:
            print(f"Skipping {name}, since no data")
            continue

        pct_lead_lowest = lead_lowest_count / total

        print(
            f"{name:20} lead lowest large diff pct {pct_lead_lowest:5.2f} total {total}"
        )

    print("")
