import pytest
import CEO.cards.deck as deck
from CEO.cards.hand import *
import CEO.cards.round as rd
import CEO.cards.player as player
from gym_ceo.envs.seat_ceo_env import SeatCEOEnv


def test_SeatCEOEnv():
    """
    Test the environment that models a player in the the CEO seat.
    """

    # Create CardValue objects for ease of use later
    cv0 = CardValue(0)
    cv1 = CardValue(1)
    cv2 = CardValue(2)
    cv3 = CardValue(3)
    cv4 = CardValue(4)
    cv5 = CardValue(5)
    cv6 = CardValue(6)
    cv7 = CardValue(7)
    cv8 = CardValue(8)

    # Create the environment
    env = SeatCEOEnv()

    env.reset()
